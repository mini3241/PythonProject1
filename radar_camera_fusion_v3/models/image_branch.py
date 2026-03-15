"""
Image branch with ResNet34 feature extraction and LSS for BEV generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional

from ..config.base import BaseConfig


class ResNet34Backbone(nn.Module):
    """ResNet34 backbone for feature extraction."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Load pre-trained ResNet34
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=config.resnet_pretrained)

        # Remove classification head
        self.feature_extractor = nn.Sequential(
            *list(self.resnet.children())[:-2]  # Remove avgpool and fc
        )

        # Feature dimension
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            features: (B, 512, H/32, W/32) feature maps
        """
        return self.feature_extractor(x)


class DepthEstimator(nn.Module):
    """
    Uncertainty-aware depth estimation module (Paper III.B.1).
    Outputs predicted depth d_{u,v} and logarithmic variance s_{u,v}
    for heteroscedastic uncertainty modeling.
    """

    def __init__(self, config: BaseConfig, max_depth: float = 75.0):
        super().__init__()
        self.config = config
        self.max_depth = max_depth

        # Shared decoder with progressive upsampling
        # Input: (B, 512, H/32, W/32) from ResNet34
        # Output: (B, 64, H/4, W/4) shared features
        self.shared_decoder = nn.Sequential(
            # First upsample: H/32 -> H/16
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Second upsample: H/16 -> H/8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Third upsample: H/8 -> H/4
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Depth prediction head: outputs d_{u,v}
        self.depth_head = nn.Conv2d(32, 1, 1)

        # Uncertainty prediction head: outputs log variance s_{u,v}
        self.log_var_head = nn.Conv2d(32, 1, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, 512, H/32, W/32) feature maps from ResNet34
        Returns:
            dict with:
                'depth': (B, 1, H/4, W/4) depth in [0, max_depth]
                'log_var': (B, 1, H/4, W/4) log variance s_{u,v}
        """
        shared_feat = self.shared_decoder(features)

        # Predicted depth
        depth = torch.sigmoid(self.depth_head(shared_feat)) * self.max_depth

        # Log variance (unbounded, represents aleatoric uncertainty)
        log_var = self.log_var_head(shared_feat)

        return {'depth': depth, 'log_var': log_var}


class LiftSplatShoot(nn.Module):
    """
    Depth-aware BEV projection following the Lift-Splat-Shoot paradigm.

    For each pixel in the feature map, predict a discrete depth distribution,
    then scatter the depth-weighted features into the BEV grid.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # BEV grid parameters
        self.x_min, self.x_max = config.bev_x_range
        self.y_min, self.y_max = config.bev_y_range
        self.resolution = config.bev_resolution

        # BEV dimensions
        self.bev_height = config.bev_height
        self.bev_width = config.bev_width

        # Depth discretization: divide [d_min, d_max] into D bins
        self.d_min = 1.0   # meters
        self.d_max = 75.0  # meters
        self.num_depth_bins = 64
        self.depth_bin_edges = torch.linspace(self.d_min, self.d_max, self.num_depth_bins + 1)

        # Depth distribution head: takes feature channels and predicts D-bin probabilities
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_depth_bins, 1)
        )

        # Feature compression: reduce channels before scattering to save memory
        self.feature_compress = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out_channels = 64

    def forward(self, features: torch.Tensor, depth: torch.Tensor,
                intrinsic: torch.Tensor,
                lidar_to_camera: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 512, Hf, Wf) image features from backbone
            depth: (B, 1, Hf, Wf) estimated depth (used as auxiliary guidance)
            intrinsic: (B, 3, 3) camera intrinsic matrix
            lidar_to_camera: (B, 4, 4) lidar-to-camera extrinsic matrix
        Returns:
            bev_features: (B, 64, H_bev, W_bev) BEV features
        """
        batch_size, _, feat_h, feat_w = features.shape
        device = features.device

        # --- Lift: predict depth distribution per pixel ---
        depth_logits = self.depth_head(features)  # (B, D, Hf, Wf)
        depth_probs = torch.softmax(depth_logits, dim=1)  # (B, D, Hf, Wf)

        # Compress feature channels
        feat_compressed = self.feature_compress(features)  # (B, C, Hf, Wf)
        C = feat_compressed.shape[1]

        # Outer product: feature * depth_prob -> (B, C, D, Hf, Wf)
        # This is the "lifted" feature volume
        feat_volume = feat_compressed.unsqueeze(2) * depth_probs.unsqueeze(1)

        # --- Splat: scatter lifted features into BEV grid ---
        # For each pixel (u, v) and each depth bin d, compute 3D point in lidar coords
        # then find corresponding BEV cell

        # Pixel coordinates grid
        u_coords = torch.arange(feat_w, device=device, dtype=torch.float32)
        v_coords = torch.arange(feat_h, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')  # (Hf, Wf)

        # Depth bin centers
        bin_centers = 0.5 * (self.depth_bin_edges[:-1] + self.depth_bin_edges[1:]).to(device)  # (D,)

        # Scale pixel coords from feature map space to original image space
        # features are at 1/32 resolution
        scale_factor = 32.0
        uu_img = uu * scale_factor  # (Hf, Wf)
        vv_img = vv * scale_factor  # (Hf, Wf)

        # Initialize BEV accumulator
        bev_features = torch.zeros(batch_size, C, self.bev_height, self.bev_width, device=device)

        # Camera-to-lidar transform
        camera_to_lidar = torch.inverse(lidar_to_camera)  # (B, 4, 4)

        for b in range(batch_size):
            K_inv = torch.inverse(intrinsic[b])  # (3, 3)
            T_c2l = camera_to_lidar[b]  # (4, 4)

            # Back-project all pixels at all depth bins to 3D camera coords
            # (Hf, Wf) -> flatten to (Hf*Wf,)
            uu_flat = uu_img.reshape(-1)  # (N,)
            vv_flat = vv_img.reshape(-1)  # (N,)
            ones_flat = torch.ones_like(uu_flat)

            # Homogeneous pixel coords: (3, N)
            uv1 = torch.stack([uu_flat, vv_flat, ones_flat], dim=0)

            # Ray directions in camera coords: (3, N)
            rays_cam = K_inv @ uv1  # (3, N)

            # For each depth bin, compute 3D points
            for d_idx in range(self.num_depth_bins):
                d = bin_centers[d_idx]

                # 3D points in camera coords: (3, N)
                pts_cam = rays_cam * d

                # Transform to lidar coords: (4, 4) @ (4, N)
                pts_cam_homo = torch.cat([pts_cam, ones_flat.unsqueeze(0)], dim=0)  # (4, N)
                pts_lidar = T_c2l @ pts_cam_homo  # (4, N)

                x_lidar = pts_lidar[0]  # (N,)
                y_lidar = pts_lidar[1]  # (N,)

                # Convert to BEV grid indices
                bev_x_idx = ((x_lidar - self.x_min) / self.resolution).long()
                bev_y_idx = ((y_lidar - self.y_min) / self.resolution).long()

                # Valid mask
                valid = (bev_x_idx >= 0) & (bev_x_idx < self.bev_width) & \
                        (bev_y_idx >= 0) & (bev_y_idx < self.bev_height)

                if valid.sum() == 0:
                    continue

                # Gather valid indices
                valid_bev_x = bev_x_idx[valid]  # (M,)
                valid_bev_y = bev_y_idx[valid]  # (M,)

                # Get the corresponding feature volume values: (C, D, Hf, Wf) -> (C, M)
                # valid indices map back to (Hf, Wf) spatial positions
                valid_feat = feat_volume[b, :, d_idx].reshape(C, -1)[:, valid]  # (C, M)

                # Scatter-add into BEV grid
                bev_linear_idx = valid_bev_y * self.bev_width + valid_bev_x  # (M,)
                bev_flat = bev_features[b].reshape(C, -1)  # (C, H*W)
                bev_flat.scatter_add_(1, bev_linear_idx.unsqueeze(0).expand(C, -1), valid_feat)

        return bev_features


class ImageBranch(nn.Module):
    """Complete image processing branch."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        self.resnet_backbone = ResNet34Backbone(config)
        self.depth_estimator = DepthEstimator(config)
        self.lss = LiftSplatShoot(config)

    def forward(self, images: torch.Tensor,
                intrinsic: torch.Tensor = None,
                lidar_to_camera: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, C, H, W) input images
            intrinsic: (B, 3, 3) camera intrinsic matrix
            lidar_to_camera: (B, 4, 4) lidar-to-camera extrinsic matrix
        Returns:
            dict with 'bev_features' (B, 64, H_bev, W_bev) and 'depth_map' (B, 1, Hf, Wf)
        """
        # Extract features
        features = self.resnet_backbone(images)  # (B, 512, H/32, W/32)

        # Estimate depth (returns dict with 'depth' and 'log_var')
        depth_out = self.depth_estimator(features)
        depth = depth_out['depth']      # (B, 1, H/4, W/4)
        log_var = depth_out['log_var']  # (B, 1, H/4, W/4)

        # Generate BEV features using depth-aware LSS
        if intrinsic is not None and lidar_to_camera is not None:
            bev_features = self.lss(features, depth, intrinsic, lidar_to_camera)
        else:
            raise ValueError("intrinsic and lidar_to_camera are required for LSS")

        return {'bev_features': bev_features, 'depth_map': depth, 'log_var': log_var}