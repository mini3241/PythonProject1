"""
Pseudo-LiDAR generation using YOLO for car detection and point cloud generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

from ..config.base import BaseConfig


class YOLODetector:
    """YOLO-based car detector."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = self._load_yolo_model()

    def _load_yolo_model(self):
        """Load YOLOv5 model using torch.hub from local repository."""
        import os

        try:
            # Check if local repo and weights exist
            if os.path.exists(self.config.yolo_repo_path) and os.path.exists(self.config.yolo_weights_path):
                print(f"Loading YOLOv5 from local repo: {self.config.yolo_repo_path}")
                print(f"Using weights: {self.config.yolo_weights_path}")

                model = torch.hub.load(
                    self.config.yolo_repo_path,
                    'custom',
                    path=self.config.yolo_weights_path,
                    source='local'
                )
            else:
                print(f"Local YOLOv5 repo or weights not found, loading from ultralytics")
                model = torch.hub.load(
                    'ultralytics/yolov5',
                    self.config.yolo_model_name,
                    pretrained=True,
                    trust_repo=True
                )

            model.conf = self.config.yolo_conf_threshold
            model.iou = self.config.yolo_iou_threshold
            model.classes = self.config.yolo_classes

            # Freeze YOLO parameters (use pretrained weights only)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            return model

        except Exception as e:
            print(f"YOLOv5 loading failed: {e}")
            print("Using dummy detector")
            return None

    def detect(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Detect cars in images.
        Args:
            images: (B, C, H, W) tensor in [0, 1] range
        Returns:
            List of detection results per image
        """
        if self.model is None:
            # Dummy detections for testing
            return self._dummy_detections(images)

        # Convert to numpy and scale to [0, 255] for YOLO
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        detections = []

        for img in images_np:
            results = self.model(img)
            dets = []
            for *xyxy, conf, cls in results.xyxy[0]:
                dets.append({
                    'bbox': xyxy,
                    'confidence': conf.item(),
                    'class': int(cls.item()),
                    'class_name': 'car'
                })
            detections.append(dets)

        return detections

    def _dummy_detections(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate dummy detections for testing."""
        batch_size = images.size(0)
        detections = []

        for _ in range(batch_size):
            # Random car detection
            detections.append([{
                'bbox': [100, 100, 200, 200],
                'confidence': 0.8,
                'class': 2,
                'class_name': 'car'
            }])

        return detections


class PointCloudGenerator:
    """Generate pseudo-point clouds from 2D detections using depth prediction."""

    def __init__(self, config: BaseConfig):
        self.config = config
        # Default camera intrinsics (can be overridden)
        self.fx = 1000.0
        self.fy = 1000.0
        self.cx = 128.0
        self.cy = 128.0

    def generate_points(self, detections: List[Dict[str, Any]],
                       depth_map: torch.Tensor,
                       intrinsic_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate pseudo-point clouds from car detections using depth prediction.
        Args:
            detections: List of detection results per image
            depth_map: (B, 1, H, W) predicted depth map
            intrinsic_matrix: (B, 3, 3) or (3, 3) camera intrinsic matrix
        Returns:
            points: (N, 5) pseudo-point cloud [x, y, z, doppler, snr]
        """
        if intrinsic_matrix is not None:
            # Handle batched intrinsic matrix (B, 3, 3) -> use first batch
            if intrinsic_matrix.dim() == 3:
                intrinsic_matrix = intrinsic_matrix[0]

            self.fx = intrinsic_matrix[0, 0].item()
            self.fy = intrinsic_matrix[1, 1].item()
            self.cx = intrinsic_matrix[0, 2].item()
            self.cy = intrinsic_matrix[1, 2].item()

        all_points = []
        total_dets = 0
        car_dets = 0

        for batch_idx, dets in enumerate(detections):
            total_dets += len(dets)
            for det in dets:
                # YOLO class 2 is car in COCO dataset
                if det['class'] == 2:  # car class
                    car_dets += 1
                    points = self._generate_car_points_from_depth(
                        det, depth_map[batch_idx, 0]
                    )
                    if len(points) > 0:
                        all_points.append(points)
                    else:
                        print(f"[DEBUG] Car detection produced 0 points. BBox: {det['bbox']}, Depth map shape: {depth_map.shape}")

        if total_dets > 0 and car_dets == 0:
            print(f"[DEBUG] {total_dets} detections but 0 cars (class 2). Classes detected: {[det['class'] for dets in detections for det in dets]}")

        if all_points:
            return torch.cat(all_points, dim=0)
        else:
            return torch.zeros((0, 5), device=depth_map.device)

    def _generate_car_points_from_depth(self, detection: Dict[str, Any],
                                       depth_map: torch.Tensor) -> torch.Tensor:
        """
        Generate points for a single car detection using depth back-projection.
        Args:
            detection: Detection dict with 'bbox' key [x1, y1, x2, y2]
            depth_map: (H, W) depth map
        Returns:
            points: (N, 5) [x, y, z, doppler, snr]
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = [float(coord) for coord in bbox]

        # Get depth map dimensions
        H, W = depth_map.shape

        # Note: bbox coordinates are in image space, need to scale to depth map space
        # Assuming depth map is downsampled from image, we need the original image size
        # For now, we'll work directly with the bbox coordinates and clamp to depth map bounds

        # Clamp bbox to depth map bounds
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            print(f"[DEBUG] Invalid bbox after clamping: [{x1}, {y1}, {x2}, {y2}]")
            return torch.zeros((0, 5), device=depth_map.device)

        # Sample points within bbox
        num_points_x = min(10, int(x2 - x1))
        num_points_y = min(10, int(y2 - y1))

        u_coords = torch.linspace(x1, x2, num_points_x, device=depth_map.device)
        v_coords = torch.linspace(y1, y2, num_points_y, device=depth_map.device)

        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='xy')
        uu = uu.flatten()
        vv = vv.flatten()

        # Get depth values at sampled points
        uu_int = uu.long().clamp(0, W - 1)
        vv_int = vv.long().clamp(0, H - 1)
        depths = depth_map[vv_int, uu_int]

        # Debug: check depth statistics
        valid_depths = depths[depths > 0.5]
        if len(valid_depths) == 0:
            print(f"[DEBUG] No valid depths (>0.5) in bbox. Depth range: [{depths.min():.4f}, {depths.max():.4f}]")
            return torch.zeros((0, 5), device=depth_map.device)

        # Back-project to 3D using camera intrinsics
        # Camera coordinate system: X-right, Y-down, Z-forward
        X_cam = (uu - self.cx) * depths / self.fx
        Y_cam = (vv - self.cy) * depths / self.fy
        Z_cam = depths

        # Convert to radar coordinate system: X-forward, Y-left, Z-up
        # Assuming camera looks forward along radar X-axis
        x_radar = Z_cam  # Camera Z -> Radar X (forward)
        y_radar = -X_cam  # Camera X -> Radar -Y (left)
        z_radar = -Y_cam  # Camera Y -> Radar -Z (up, assuming camera Y is down)

        # Add dummy doppler and SNR values
        doppler = torch.zeros_like(x_radar)
        snr = torch.ones_like(x_radar) * 10.0  # Default SNR

        # Stack to (N, 5)
        points = torch.stack([x_radar, y_radar, z_radar, doppler, snr], dim=1)

        # Filter out invalid points
        # x_radar = Z_cam (forward), y_radar = -X_cam (left-right)
        # x_radar corresponds to bev_y_range (forward distance)
        # y_radar corresponds to bev_x_range (lateral distance)
        valid_mask = (depths > 0.5) & \
                     (x_radar > self.config.bev_y_range[0]) & \
                     (x_radar < self.config.bev_y_range[1]) & \
                     (y_radar > self.config.bev_x_range[0]) & \
                     (y_radar < self.config.bev_x_range[1])

        points_filtered = points[valid_mask]

        if len(points_filtered) == 0:
            # Debug: why were all points filtered?
            depth_valid = (depths > 0.1).sum().item()
            x_valid = ((x_radar > self.config.bev_y_range[0]) & (x_radar < self.config.bev_y_range[1])).sum().item()
            y_valid = ((y_radar > self.config.bev_x_range[0]) & (y_radar < self.config.bev_x_range[1])).sum().item()
            print(f"[DEBUG] All points filtered. Valid: depth={depth_valid}/{len(depths)}, "
                  f"x_range={x_valid}/{len(depths)}, y_range={y_valid}/{len(depths)}")
            print(f"[DEBUG] x_radar range: [{x_radar.min():.2f}, {x_radar.max():.2f}], "
                  f"BEV y_range: {self.config.bev_y_range}")
            print(f"[DEBUG] y_radar range: [{y_radar.min():.2f}, {y_radar.max():.2f}], "
                  f"BEV x_range: {self.config.bev_x_range}")

        return points_filtered


class PseudoLidarBranch(nn.Module):
    """Complete pseudo-LiDAR processing branch."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        self.detector = YOLODetector(config)
        self.point_generator = PointCloudGenerator(config)
        # Reuse radar branch for point cloud processing
        from .radar_branch import RadarBranch
        self.radar_processor = RadarBranch(config)

    def forward(self, images: torch.Tensor,
               depth_map: torch.Tensor,
               intrinsic_matrix: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Args:
            images: (B, C, H, W) input images
            depth_map: (B, 1, H_d, W_d) predicted depth map (may be downsampled)
            intrinsic_matrix: (3, 3) camera intrinsic matrix
        Returns:
            Dictionary containing:
                - 'bev_features': (B, 128, H, W) BEV features
                - 'yolo_detections': List of detection results per image
                - 'pseudo_points': (N, 5) generated pseudo-point cloud
        """
        # Detect cars
        detections = self.detector.detect(images)

        # Upsample depth map to match image size for bbox coordinate alignment
        B, C, H_img, W_img = images.shape
        _, _, H_depth, W_depth = depth_map.shape

        if H_depth != H_img or W_depth != W_img:
            depth_map_upsampled = F.interpolate(
                depth_map,
                size=(H_img, W_img),
                mode='bilinear',
                align_corners=False
            )
        else:
            depth_map_upsampled = depth_map

        # Generate pseudo-point clouds using upsampled depth
        points = self.point_generator.generate_points(detections, depth_map_upsampled, intrinsic_matrix)

        batch_size = images.size(0)

        if len(points) == 0:
            # No detections, return empty BEV
            return {
                'bev_features': torch.zeros(
                    batch_size, 128, self.config.bev_height, self.config.bev_width,
                    device=images.device
                ),
                'yolo_detections': detections,
                'pseudo_points': points
            }

        # Process through radar branch
        radar_data = {'points': points}
        bev_features = self.radar_processor(radar_data)

        # Ensure correct batch size by expanding if needed
        if bev_features.size(0) != batch_size:
            # Repeat the single BEV feature map for all batch samples
            bev_features = bev_features.repeat(batch_size, 1, 1, 1)

        return {
            'bev_features': bev_features,
            'yolo_detections': detections,
            'pseudo_points': points
        }