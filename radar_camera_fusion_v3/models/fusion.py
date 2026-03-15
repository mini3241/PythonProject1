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
    Cross-Modal Transformer (Paper III.D / Fig.1).

    1. Pseudo-point BEV and Radar BEV are pre-fused into a unified
       geometric representation (Keys & Values).
    2. Image BEV provides the semantic Queries.
    3. Multi-head cross-attention aligns semantics with geometry.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8,
                 attn_spatial_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_spatial_size = attn_spatial_size

        # --- Modality projections to common embed_dim ---
        # Image (64ch) -> Query
        self.image_proj = nn.Sequential(
            nn.Conv2d(64, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Radar (128ch) -> part of Key/Value
        self.radar_proj = nn.Sequential(
            nn.Conv2d(128, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Pseudo-point (128ch) -> part of Key/Value
        self.pseudo_proj = nn.Sequential(
            nn.Conv2d(128, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # --- Pre-fusion: merge pseudo + radar into geometric BEV ---
        # Concatenate (2*embed_dim) -> compress to embed_dim
        self.geo_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # --- Learnable positional encodings (Paper III.D.1 Epos) ---
        self.query_pos = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
        self.kv_pos = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        # --- Cross-attention: Q=image, K/V=geometric ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # --- Post-attention: LayerNorm + FFN ---
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # --- Output projection ---
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_bev: torch.Tensor,
                radar_bev: torch.Tensor,
                pseudo_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_bev:  (B, 64,  H, W)  - semantic features (Query)
            radar_bev:  (B, 128, H, W)  - geometric features
            pseudo_bev: (B, 128, H, W)  - depth-aware geometric features
        Returns:
            fused: (B, 128, H, W)
        """
        B, _, H, W = image_bev.shape
        S = self.attn_spatial_size

        # --- Step 1: Project each modality to embed_dim ---
        img_feat = self.image_proj(image_bev)     # (B, D, H, W)
        rad_feat = self.radar_proj(radar_bev)     # (B, D, H, W)
        pse_feat = self.pseudo_proj(pseudo_bev)   # (B, D, H, W)

        # --- Step 2: Pre-fuse pseudo + radar into geometric BEV (K/V) ---
        geo_cat = torch.cat([rad_feat, pse_feat], dim=1)  # (B, 2D, H, W)
        geo_feat = self.geo_fusion(geo_cat)                # (B, D, H, W)

        # --- Step 3: Add positional encodings ---
        query_feat = img_feat + self.query_pos   # (B, D, H, W)
        kv_feat = geo_feat + self.kv_pos         # (B, D, H, W)

        # --- Step 4: Downsample for attention efficiency ---
        query_ds = F.adaptive_avg_pool2d(query_feat, (S, S))  # (B, D, S, S)
        kv_ds = F.adaptive_avg_pool2d(kv_feat, (S, S))        # (B, D, S, S)

        N = S * S

        # Flatten to sequences: (B, N, D)
        query_seq = query_ds.flatten(2).permute(0, 2, 1)  # (B, N, D)
        kv_seq = kv_ds.flatten(2).permute(0, 2, 1)        # (B, N, D)

        # --- Step 5: Cross-attention (Q=image, K=V=geometric) ---
        attn_out, _ = self.cross_attention(query_seq, kv_seq, kv_seq)

        # Residual + LayerNorm + FFN
        attn_out = self.norm1(query_seq + attn_out)
        attn_out = self.norm2(attn_out + self.ffn(attn_out))

        # --- Step 6: Reshape back to spatial and upsample ---
        attn_spatial = attn_out.permute(0, 2, 1).reshape(B, self.embed_dim, S, S)
        attn_spatial = F.interpolate(attn_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # --- Step 7: Output projection ---
        fused = self.output_proj(attn_spatial)  # (B, D, H, W)

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
                image_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radar_bev:  (B, 128, H, W)
            pseudo_bev: (B, 128, H, W)
            image_bev:  (B, 64, H, W)
        Returns:
            fused: (B, 128, H, W)
        """
        return self.cmt(image_bev, radar_bev, pseudo_bev)
