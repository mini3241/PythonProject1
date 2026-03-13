"""
Base model that integrates radar, pseudo-LiDAR, and image branches with fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..config.base import BaseConfig
from .radar_branch import RadarBranch
from .pseudo_lidar import PseudoLidarBranch
from .image_branch import ImageBranch
from .fusion import FusionModule


class RadarCameraFusionModel(nn.Module):
    """Complete radar-camera fusion model."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Initialize branches
        self.radar_branch = RadarBranch(config)
        self.pseudo_lidar_branch = PseudoLidarBranch(config)
        self.image_branch = ImageBranch(config)

        # Fusion module
        self.fusion_module = FusionModule(config)

        # Optional detection/tracking head
        self.detection_head = self._build_detection_head()

    def _build_detection_head(self) -> Optional[nn.Module]:
        """Build detection head if needed."""
        if hasattr(self.config, 'enable_detection') and self.config.enable_detection:
            # CMT fusion output: 128 channels
            in_channels = 128

            return nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1)  # Output heatmap
            )
        return None

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through all branches.
        Args:
            data: Dictionary containing:
                - 'radar_points': radar points [x, y, z, doppler, snr]
                - 'images': (B, C, H, W) camera images
                - 'intrinsic_matrix': (B, 3, 3) camera intrinsics
                - 'lidar_to_camera_extrinsic': (B, 4, 4) extrinsic matrix
        Returns:
            Dictionary with outputs from all branches and fusion
        """
        outputs = {}

        intrinsic_matrix = data.get('intrinsic_matrix')
        lidar_to_camera = data.get('lidar_to_camera_extrinsic')

        # Process radar branch
        if 'radar_points' in data:
            radar_data = {'points': data['radar_points']}
            outputs['radar_bev'] = self.radar_branch(radar_data)

        # Process image branch (returns dict with bev_features + depth_map)
        depth_map = None
        if 'images' in data:
            if intrinsic_matrix is None or lidar_to_camera is None:
                raise ValueError("intrinsic_matrix and lidar_to_camera_extrinsic are required")

            image_out = self.image_branch(
                data['images'], intrinsic_matrix, lidar_to_camera
            )
            outputs['image_bev'] = image_out['bev_features']   # (B, 64, H, W)
            depth_map = image_out['depth_map']                  # (B, 1, Hf, Wf)
            outputs['depth_map'] = depth_map

        # Process pseudo-LiDAR branch using depth from image branch
        if 'images' in data and depth_map is not None:
            pseudo_outputs = self.pseudo_lidar_branch(
                data['images'], depth_map, intrinsic_matrix
            )
            outputs['pseudo_bev'] = pseudo_outputs['bev_features']
            outputs['yolo_detections'] = pseudo_outputs['yolo_detections']
            outputs['pseudo_points'] = pseudo_outputs['pseudo_points']

        # Fuse features
        if all(k in outputs for k in ['radar_bev', 'pseudo_bev', 'image_bev']):
            outputs['fused_bev'] = self.fusion_module(
                outputs['radar_bev'],
                outputs['pseudo_bev'],
                outputs['image_bev']
            )

            # Apply detection head if available
            if self.detection_head is not None:
                outputs['detection_map'] = self.detection_head(outputs['fused_bev'])

        return outputs

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for each component."""
        return {
            'radar_branch': sum(p.numel() for p in self.radar_branch.parameters()),
            'pseudo_lidar_branch': sum(p.numel() for p in self.pseudo_lidar_branch.parameters()),
            'image_branch': sum(p.numel() for p in self.image_branch.parameters()),
            'fusion_module': sum(p.numel() for p in self.fusion_module.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
