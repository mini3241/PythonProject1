"""
Radar branch for processing mmWave radar data.
Uses simplified PointPillars-like structure to generate BEV features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from ..config.base import BaseConfig


class VoxelFeatureEncoder(nn.Module):
    """Voxel feature encoder for radar points."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Feature encoding layers
        self.mlp = nn.Sequential(
            nn.Linear(len(config.radar_features), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (N, 5) tensor with [x, y, z, doppler, snr]
        Returns:
            features: (N, 64) encoded features
        """
        if points.size(0) <= 1 and self.training:
            # BatchNorm requires >1 samples; temporarily use eval mode
            self.mlp.eval()
            out = self.mlp(points)
            self.mlp.train()
            return out
        return self.mlp(points)


class PillarScatter(nn.Module):
    """Scatter voxel features to BEV grid."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    def forward(self, voxel_coords: torch.Tensor, voxel_features: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """
        Args:
            voxel_coords: (M, 3) [batch_idx, y_idx, x_idx]
            voxel_features: (M, C) voxel features
            batch_size: Optional explicit batch size
        Returns:
            bev_features: (B, C, H, W)
        """
        if batch_size is None:
            batch_size = voxel_coords[:, 0].max().item() + 1 if len(voxel_coords) > 0 else 1
        bev_features = torch.zeros(
            batch_size,
            voxel_features.size(1),
            self.config.bev_height,
            self.config.bev_width,
            device=voxel_features.device
        )

        # Scatter features to BEV grid
        for i in range(batch_size):
            mask = voxel_coords[:, 0] == i
            if mask.any():
                coords = voxel_coords[mask, 1:3].long()
                features = voxel_features[mask]
                bev_features[i, :, coords[:, 0], coords[:, 1]] = features.t()

        return bev_features


class RadarBackbone(nn.Module):
    """Backbone network for radar BEV features."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Input: (B, 64, H, W)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RadarBranch(nn.Module):
    """Complete radar processing branch."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        self.voxel_encoder = VoxelFeatureEncoder(config)
        self.pillar_scatter = PillarScatter(config)
        self.backbone = RadarBackbone(config)

    def voxelize(self, points: torch.Tensor) -> tuple:
        """
        Voxelize radar points.
        Args:
            points: (N, 5) [x, y, z, doppler, snr]
        Returns:
            voxel_coords: (M, 3) [batch_idx, y, x]
            voxel_features: (M, C)
        """
        # Convert to voxel coordinates
        voxel_x = ((points[:, 0] - self.config.bev_x_range[0]) /
                  (self.config.bev_x_range[1] - self.config.bev_x_range[0]) *
                  self.config.bev_width).clamp(0, self.config.bev_width - 1)
        voxel_y = ((points[:, 1] - self.config.bev_y_range[0]) /
                  (self.config.bev_y_range[1] - self.config.bev_y_range[0]) *
                  self.config.bev_height).clamp(0, self.config.bev_height - 1)

        # Simple voxelization (for demo - can be enhanced)
        voxel_coords = torch.stack([voxel_x, voxel_y], dim=1).long()

        # Encode features
        voxel_features = self.voxel_encoder(points)

        return voxel_coords, voxel_features

    def forward(self, radar_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            radar_data: dict with 'points' key containing (N, 5) tensor or list of tensors
        Returns:
            bev_features: (B, 128, H, W)
        """
        points = radar_data['points']

        # Handle both single tensor and list of tensors
        if isinstance(points, list):
            # Batch processing
            batch_size = len(points)
            all_voxel_coords = []
            all_voxel_features = []

            for batch_idx, pts in enumerate(points):
                if len(pts) == 0:
                    continue
                # Voxelize each sample
                voxel_coords, voxel_features = self.voxelize(pts)

                # Add batch index
                batch_coords = torch.cat([
                    torch.full((len(voxel_coords), 1), batch_idx, device=pts.device, dtype=torch.long),
                    voxel_coords
                ], dim=1)

                all_voxel_coords.append(batch_coords)
                all_voxel_features.append(voxel_features)

            if len(all_voxel_coords) > 0:
                # Concatenate all batches
                batch_coords = torch.cat(all_voxel_coords, dim=0)
                voxel_features = torch.cat(all_voxel_features, dim=0)
            else:
                # Empty batch
                device = points[0].device if len(points) > 0 else torch.device('cuda')
                batch_coords = torch.zeros((0, 3), device=device, dtype=torch.long)
                voxel_features = torch.zeros((0, 128), device=device)

            # Scatter to BEV
            bev = self.pillar_scatter(batch_coords, voxel_features, batch_size=batch_size)
        else:
            # Single sample processing (batch_size=1)
            # Voxelize
            voxel_coords, voxel_features = self.voxelize(points)

            # Add batch dimension
            batch_coords = torch.cat([
                torch.zeros(len(voxel_coords), 1, device=points.device, dtype=torch.long),
                voxel_coords
            ], dim=1)

            # Scatter to BEV
            bev = self.pillar_scatter(batch_coords, voxel_features)

        # Process through backbone
        bev_features = self.backbone(bev)

        return bev_features