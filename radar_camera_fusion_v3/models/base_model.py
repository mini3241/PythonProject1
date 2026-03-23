"""
融合模型主体，集成雷达、伪LiDAR、图像分支 + 时序累积TCA。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from collections import deque

from ..config.base import BaseConfig
from .radar_branch import RadarBranch
from .pseudo_lidar import PseudoLidarBranch
from .image_branch import ImageBranch
from .fusion import FusionModule


class RadarCameraFusionModel(nn.Module):
    """完整的雷达-相机融合模型，含时序累积TCA（论文公式7）。"""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # 各分支
        self.radar_branch = RadarBranch(config)
        self.pseudo_lidar_branch = PseudoLidarBranch(config)
        self.image_branch = ImageBranch(config)

        # 融合模块
        self.fusion_module = FusionModule(config)

        # 检测头
        self.detection_head = self._build_detection_head()

        # === 时序累积TCA（论文 III.E.2 公式7） ===
        # F_final = F_spatial + Σ_{k=1}^{K} λ^k · F_final^{t-k}
        # 简化版：不做ego-motion warp（数据集无精确ego-motion），直接累积
        self.tca_K = 3            # 历史帧数
        self.tca_lambda = 0.5     # 衰减因子
        self.history_buffer = deque(maxlen=self.tca_K)
        self.confidence_buffer = deque(maxlen=self.tca_K)

    def _build_detection_head(self) -> Optional[nn.Module]:
        """构建检测头。"""
        if hasattr(self.config, 'enable_detection') and self.config.enable_detection:
            in_channels = 128
            return nn.Sequential(
                nn.Conv2d(in_channels, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1)
            )
        return None

    def reset_temporal(self):
        """重置时序缓存（场景切换时调用）。"""
        self.history_buffer.clear()
        self.confidence_buffer.clear()

    def _temporal_accumulation(self, F_spatial, confidence_bev=None):
        """
        时序累积（论文公式7）：
        F_final = F_spatial + Σ_{k=1}^{K} λ^k · F_history[k]

        同时累积置信度，用于自适应卡尔曼噪声。
        """
        F_final = F_spatial.clone()

        # 累积历史帧特征
        for k, hist_feat in enumerate(reversed(list(self.history_buffer))):
            decay = self.tca_lambda ** (k + 1)
            if hist_feat.shape == F_spatial.shape:
                F_final = F_final + decay * hist_feat

        # 保存当前帧到缓存
        self.history_buffer.append(F_spatial.detach())

        # 累积置信度
        fused_confidence = confidence_bev
        if confidence_bev is not None:
            for k, hist_conf in enumerate(reversed(list(self.confidence_buffer))):
                decay = self.tca_lambda ** (k + 1)
                if hist_conf.shape == confidence_bev.shape:
                    fused_confidence = torch.max(fused_confidence, decay * hist_conf)
            self.confidence_buffer.append(confidence_bev.detach())

        return F_final, fused_confidence

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        前向传播。
        Args:
            data: 包含 radar_points, images, intrinsic_matrix,
                  lidar_to_camera_extrinsic 的字典
        Returns:
            包含各分支输出和融合结果的字典
        """
        outputs = {}

        intrinsic_matrix = data.get('intrinsic_matrix')
        lidar_to_camera = data.get('lidar_to_camera_extrinsic')

        # 雷达分支
        if 'radar_points' in data:
            radar_data = {'points': data['radar_points']}
            outputs['radar_bev'] = self.radar_branch(radar_data)

        # 图像分支
        depth_map = None
        if 'images' in data:
            if intrinsic_matrix is None or lidar_to_camera is None:
                raise ValueError("需要 intrinsic_matrix 和 lidar_to_camera_extrinsic")

            image_out = self.image_branch(
                data['images'], intrinsic_matrix, lidar_to_camera
            )
            outputs['image_bev'] = image_out['bev_features']
            depth_map = image_out['depth_map']
            outputs['depth_map'] = depth_map
            outputs['log_var'] = image_out.get('log_var', None)

        # 伪LiDAR分支（含置信度加权，论文公式2）
        log_var = outputs.get('log_var', None)
        if 'images' in data and depth_map is not None:
            pseudo_outputs = self.pseudo_lidar_branch(
                data['images'], depth_map, intrinsic_matrix, log_var=log_var
            )
            outputs['pseudo_bev'] = pseudo_outputs['bev_features']
            outputs['confidence_bev'] = pseudo_outputs.get('confidence_bev', None)
            outputs['yolo_detections'] = pseudo_outputs['yolo_detections']
            outputs['pseudo_points'] = pseudo_outputs['pseudo_points']

        # 融合（CMT交叉注意力 + VCW，论文公式5/6）
        if all(k in outputs for k in ['radar_bev', 'pseudo_bev', 'image_bev']):
            confidence_bev = outputs.get('confidence_bev', None)
            F_spatial = self.fusion_module(
                outputs['radar_bev'],
                outputs['pseudo_bev'],
                outputs['image_bev'],
                confidence_bev=confidence_bev
            )

            # 时序累积TCA（论文公式7）
            if not self.training:
                # 推理时启用时序累积
                F_final, fused_conf = self._temporal_accumulation(
                    F_spatial, confidence_bev
                )
                outputs['fused_bev'] = F_final
                outputs['fused_confidence'] = fused_conf
            else:
                outputs['fused_bev'] = F_spatial

            # 检测头
            if self.detection_head is not None:
                outputs['detection_map'] = self.detection_head(outputs['fused_bev'])

        return outputs

    def get_parameter_count(self) -> Dict[str, int]:
        """各组件参数量统计。"""
        return {
            'radar_branch': sum(p.numel() for p in self.radar_branch.parameters()),
            'pseudo_lidar_branch': sum(p.numel() for p in self.pseudo_lidar_branch.parameters()),
            'image_branch': sum(p.numel() for p in self.image_branch.parameters()),
            'fusion_module': sum(p.numel() for p in self.fusion_module.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
