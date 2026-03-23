"""
Cross-Modal Transformer (CMT) fusion module.

Paper III.D / Fig.1: Image BEV features serve as Queries, while
pseudo-point BEV and radar BEV are pre-fused to form geometric Keys
and Values for cross-attention, enforcing semantic-geometric alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from ..config.base import BaseConfig


class CrossModalTransformer(nn.Module):
    """
    跨模态Transformer（论文 III.D）+ 体素级置信度加权VCW（论文 III.E.1 公式6）。

    1. 伪点云BEV和雷达BEV预融合为几何表示（Keys & Values）
    2. 图像BEV提供语义Queries
    3. 多头交叉注意力对齐语义与几何
    4. VCW：F_spatial = α⊙F_img + (1-α)⊙F_radar + F_att
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8,
                 attn_spatial_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_spatial_size = attn_spatial_size

        # 各模态投影到统一维度
        self.image_proj = nn.Sequential(
            nn.Conv2d(64, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.radar_proj = nn.Sequential(
            nn.Conv2d(128, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pseudo_proj = nn.Sequential(
            nn.Conv2d(128, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 预融合：伪点云 + 雷达 -> 几何BEV
        self.geo_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 可学习位置编码（论文 III.D.1 Epos）
        self.query_pos = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
        self.kv_pos = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        # 交叉注意力：Q=图像, K/V=几何
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # 后处理：LayerNorm + FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # VCW: 从置信度图生成融合权重α（论文公式6）
        # 输入: 置信度BEV(1ch) + 雷达占据密度(1ch) -> α(1ch)
        self.alpha_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_bev: torch.Tensor,
                radar_bev: torch.Tensor,
                pseudo_bev: torch.Tensor,
                confidence_bev: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            image_bev:  (B, 64,  H, W)  语义特征 (Query)
            radar_bev:  (B, 128, H, W)  几何特征
            pseudo_bev: (B, 128, H, W)  深度感知几何特征
            confidence_bev: (B, 1, H, W) 伪点云置信度图
        Returns:
            fused: (B, 128, H, W)
        """
        B, _, H, W = image_bev.shape
        S = self.attn_spatial_size

        # Step 1: 投影到统一维度
        img_feat = self.image_proj(image_bev)     # (B, D, H, W)
        rad_feat = self.radar_proj(radar_bev)     # (B, D, H, W)
        pse_feat = self.pseudo_proj(pseudo_bev)   # (B, D, H, W)

        # Step 2: 预融合伪点云+雷达 -> 几何BEV (K/V)
        geo_cat = torch.cat([rad_feat, pse_feat], dim=1)
        geo_feat = self.geo_fusion(geo_cat)

        # Step 3: 添加位置编码
        query_feat = img_feat + self.query_pos
        kv_feat = geo_feat + self.kv_pos

        # Step 4: 下采样后做注意力
        query_ds = F.adaptive_avg_pool2d(query_feat, (S, S))
        kv_ds = F.adaptive_avg_pool2d(kv_feat, (S, S))

        N = S * S
        query_seq = query_ds.flatten(2).permute(0, 2, 1)
        kv_seq = kv_ds.flatten(2).permute(0, 2, 1)

        # Step 5: 交叉注意力 (论文公式5)
        attn_out, _ = self.cross_attention(query_seq, kv_seq, kv_seq)
        attn_out = self.norm1(query_seq + attn_out)
        attn_out = self.norm2(attn_out + self.ffn(attn_out))

        # Step 6: 还原空间维度
        attn_spatial = attn_out.permute(0, 2, 1).reshape(B, self.embed_dim, S, S)
        F_att = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # Step 7: VCW体素级置信度加权（论文公式6）
        # F_spatial = α⊙F_img + (1-α)⊙F_radar + F_att
        if confidence_bev is not None:
            # 对齐confidence_bev到当前BEV特征的空间尺寸和batch
            if confidence_bev.shape[2:] != (H, W):
                confidence_bev = F.interpolate(confidence_bev, size=(H, W), mode='bilinear', align_corners=False)
            if confidence_bev.shape[0] != B:
                confidence_bev = confidence_bev[:1].expand(B, -1, -1, -1)
            radar_occupancy = (radar_bev.abs().sum(dim=1, keepdim=True) > 0).float()
            alpha_input = torch.cat([confidence_bev, radar_occupancy], dim=1)
            alpha = self.alpha_net(alpha_input)  # (B, 1, H, W)
        else:
            alpha = torch.full((B, 1, H, W), 0.5, device=image_bev.device)

        F_spatial = alpha * img_feat + (1 - alpha) * rad_feat + F_att

        fused = self.output_proj(F_spatial)
        return fused


class FusionModule(nn.Module):
    """
    Main fusion module.

    Wraps CrossModalTransformer, keeps the same forward signature as the
    original FusionModule so base_model.py needs zero changes.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        self.cmt = CrossModalTransformer(embed_dim=128, num_heads=8)

    def forward(self, radar_bev: torch.Tensor,
                pseudo_bev: torch.Tensor,
                image_bev: torch.Tensor,
                confidence_bev: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            radar_bev:  (B, 128, H, W)
            pseudo_bev: (B, 128, H, W)
            image_bev:  (B, 64, H, W)
            confidence_bev: (B, 1, H, W) 伪点云置信度图
        Returns:
            fused: (B, 128, H, W)
        """
        return self.cmt(image_bev, radar_bev, pseudo_bev, confidence_bev)
