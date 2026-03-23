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
        """从本地仓库加载YOLOv5模型。"""
        import os

        try:
            if os.path.exists(self.config.yolo_repo_path) and os.path.exists(self.config.yolo_weights_path):
                print(f"从本地加载YOLOv5: {self.config.yolo_repo_path}")
                print(f"权重文件: {self.config.yolo_weights_path}")

                model = torch.hub.load(
                    self.config.yolo_repo_path,
                    'custom',
                    path=self.config.yolo_weights_path,
                    source='local',
                    force_reload=True
                )
            else:
                print(f"本地YOLOv5未找到，从ultralytics下载")
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
                       intrinsic_matrix: Optional[torch.Tensor] = None,
                       log_var_map: Optional[torch.Tensor] = None,
                       beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Generate confidence-weighted pseudo-point clouds (Paper III.B.2, Eq.2).
        Args:
            detections: List of detection results per image
            depth_map: (B, 1, H, W) predicted depth map
            intrinsic_matrix: (B, 3, 3) or (3, 3) camera intrinsic matrix
            log_var_map: (B, 1, H, W) pixel-level log variance (uncertainty)
            beta: scale control parameter for confidence weighting
        Returns:
            dict with 'points' (N, 5) and 'confidence_weights' (N,)
        """
        if intrinsic_matrix is not None:
            if intrinsic_matrix.dim() == 3:
                intrinsic_matrix = intrinsic_matrix[0]

            self.fx = intrinsic_matrix[0, 0].item()
            self.fy = intrinsic_matrix[1, 1].item()
            self.cx = intrinsic_matrix[0, 2].item()
            self.cy = intrinsic_matrix[1, 2].item()

        all_points = []
        all_weights = []
        total_dets = 0
        car_dets = 0

        for batch_idx, dets in enumerate(detections):
            total_dets += len(dets)
            lv_map = log_var_map[batch_idx, 0] if log_var_map is not None else None
            for det in dets:
                if det['class'] == 2:
                    car_dets += 1
                    points, weights = self._generate_car_points_from_depth(
                        det, depth_map[batch_idx, 0], lv_map, beta
                    )
                    if len(points) > 0:
                        all_points.append(points)
                        all_weights.append(weights)

        if all_points:
            return {
                'points': torch.cat(all_points, dim=0),
                'confidence_weights': torch.cat(all_weights, dim=0)
            }
        else:
            return {
                'points': torch.zeros((0, 5), device=depth_map.device),
                'confidence_weights': torch.zeros((0,), device=depth_map.device)
            }

    def _generate_car_points_from_depth(self, detection: Dict[str, Any],
                                       depth_map: torch.Tensor,
                                       log_var_map: Optional[torch.Tensor] = None,
                                       beta: float = 1.0) -> tuple:
        """
        Generate confidence-weighted points for a single car detection (Paper Eq.2).
        Args:
            detection: Detection dict with 'bbox' key [x1, y1, x2, y2]
            depth_map: (H, W) depth map
            log_var_map: (H, W) log variance map for uncertainty
            beta: scale control parameter
        Returns:
            points: (N, 5) [x, y, z, doppler, snr]
            weights: (N,) confidence weights w_k = exp(-beta * log_var)
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = [float(coord) for coord in bbox]

        H, W = depth_map.shape

        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 <= x1 or y2 <= y1:
            return (torch.zeros((0, 5), device=depth_map.device),
                    torch.zeros((0,), device=depth_map.device))

        num_points_x = min(10, int(x2 - x1))
        num_points_y = min(10, int(y2 - y1))

        u_coords = torch.linspace(x1, x2, num_points_x, device=depth_map.device)
        v_coords = torch.linspace(y1, y2, num_points_y, device=depth_map.device)

        uu, vv = torch.meshgrid(u_coords, v_coords, indexing='xy')
        uu = uu.flatten()
        vv = vv.flatten()

        uu_int = uu.long().clamp(0, W - 1)
        vv_int = vv.long().clamp(0, H - 1)
        depths = depth_map[vv_int, uu_int]

        # Compute confidence weights (Paper Eq.2): w_k = exp(-β · s_{u,v})
        if log_var_map is not None:
            log_vars = log_var_map[vv_int, uu_int]
            conf_weights = torch.exp(-beta * log_vars)
            conf_weights = conf_weights.clamp(0.0, 1.0)
        else:
            conf_weights = torch.ones_like(depths)

        valid_depths = depths[depths > 0.5]
        if len(valid_depths) == 0:
            return (torch.zeros((0, 5), device=depth_map.device),
                    torch.zeros((0,), device=depth_map.device))

        X_cam = (uu - self.cx) * depths / self.fx
        Y_cam = (vv - self.cy) * depths / self.fy
        Z_cam = depths

        x_radar = Z_cam
        y_radar = -X_cam
        z_radar = -Y_cam

        doppler = torch.zeros_like(x_radar)
        snr = torch.ones_like(x_radar) * 10.0

        points = torch.stack([x_radar, y_radar, z_radar, doppler, snr], dim=1)

        valid_mask = (depths > 0.5) & \
                     (x_radar > self.config.bev_y_range[0]) & \
                     (x_radar < self.config.bev_y_range[1]) & \
                     (y_radar > self.config.bev_x_range[0]) & \
                     (y_radar < self.config.bev_x_range[1])

        return points[valid_mask], conf_weights[valid_mask]


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
               intrinsic_matrix: Optional[torch.Tensor] = None,
               log_var: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        伪LiDAR分支前向传播，支持置信度加权（论文公式2）。
        Args:
            images: (B, C, H, W) 输入图像
            depth_map: (B, 1, H_d, W_d) 预测深度图
            intrinsic_matrix: (B, 3, 3) 相机内参
            log_var: (B, 1, H_d, W_d) 像素级对数方差（不确定性）
        Returns:
            包含 bev_features, confidence_bev, yolo_detections, pseudo_points 的字典
        """
        # YOLO检测车辆
        detections = self.detector.detect(images)

        # 上采样深度图到图像尺寸，对齐bbox坐标
        B, C, H_img, W_img = images.shape
        _, _, H_depth, W_depth = depth_map.shape

        if H_depth != H_img or W_depth != W_img:
            depth_map_upsampled = F.interpolate(
                depth_map, size=(H_img, W_img),
                mode='bilinear', align_corners=False
            )
            log_var_upsampled = F.interpolate(
                log_var, size=(H_img, W_img),
                mode='bilinear', align_corners=False
            ) if log_var is not None else None
        else:
            depth_map_upsampled = depth_map
            log_var_upsampled = log_var

        # 生成置信度加权的伪点云
        result = self.point_generator.generate_points(
            detections, depth_map_upsampled, intrinsic_matrix,
            log_var_map=log_var_upsampled, beta=1.0
        )
        points = result['points']
        conf_weights = result['confidence_weights']

        batch_size = images.size(0)
        bev_h = self.config.bev_height
        bev_w = self.config.bev_width

        if len(points) == 0:
            return {
                'bev_features': torch.zeros(batch_size, 128, bev_h, bev_w, device=images.device),
                'confidence_bev': torch.zeros(batch_size, 1, bev_h, bev_w, device=images.device),
                'yolo_detections': detections,
                'pseudo_points': points
            }

        # 通过radar分支处理伪点云 -> BEV特征
        radar_data = {'points': points}
        bev_features = self.radar_processor(radar_data)

        if bev_features.size(0) != batch_size:
            bev_features = bev_features[:1].expand(batch_size, -1, -1, -1)

        # 构建置信度BEV图（batch=1，之后expand）
        confidence_bev = torch.zeros(1, 1, bev_h, bev_w, device=images.device)
        voxel_x = ((points[:, 0] - self.config.bev_x_range[0]) /
                   (self.config.bev_x_range[1] - self.config.bev_x_range[0]) *
                   bev_w).clamp(0, bev_w - 1).long()
        voxel_y = ((points[:, 1] - self.config.bev_y_range[0]) /
                   (self.config.bev_y_range[1] - self.config.bev_y_range[0]) *
                   bev_h).clamp(0, bev_h - 1).long()

        for i in range(len(points)):
            cx, cy = voxel_x[i], voxel_y[i]
            confidence_bev[0, 0, cy, cx] = max(
                confidence_bev[0, 0, cy, cx].item(), conf_weights[i].item()
            )

        # expand到batch_size（共享同一置信度图）
        confidence_bev = confidence_bev.expand(batch_size, -1, -1, -1)

        return {
            'bev_features': bev_features,
            'confidence_bev': confidence_bev,
            'yolo_detections': detections,
            'pseudo_points': points
        }