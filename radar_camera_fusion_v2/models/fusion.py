"""
Cross-Modal Transformer (CMT) fusion module.

Paper III.D: project image, pseudo-point, and radar tokens into a
unified representation space, utilizing cross-attention to enforce
semantic and geometric alignment.

All three modalities are projected to a common embedding dimension,
concatenated along the token (spatial) axis, and processed through
self-attention so every modality can attend to every other modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from ..config.base import BaseConfig


class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer (Paper III.D).

    Projects image / pseudo-point / radar BEV features into a unified
    token space and applies multi-head self-attention across all three
    modality tokens, enabling full cross-modal interaction.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8,
                 attn_spatial_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_spatial_size = attn_spatial_size  # downsample BEV to this before attention

        # Project each modality to common embed_dim
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

        # Learnable modality-type embeddings (like segment embeddings in BERT)
        # so the attention can distinguish which tokens come from which modality
        self.image_type_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
        self.radar_type_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
        self.pseudo_type_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)

        # Multi-head self-attention over the concatenated tri-modal tokens
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Post-attention: LayerNorm + FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Output projection: merge tri-modal output back to single BEV
        # After attention each modality gets N tokens, we take all 3*N tokens
        # and compress back via a 1x1 conv over the 3-channel stack
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image_bev: torch.Tensor,
                radar_bev: torch.Tensor,
                pseudo_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_bev:  (B, 64,  H, W)
            radar_bev:  (B, 128, H, W)
            pseudo_bev: (B, 128, H, W)
        Returns:
            fused: (B, 128, H, W)
        """
        B, _, H, W = image_bev.shape
        S = self.attn_spatial_size  # attention spatial size (e.g. 32)

        # Project to common dimension + add modality-type embeddings
        img_tok = self.image_proj(image_bev) + self.image_type_embed    # (B, D, H, W)
        rad_tok = self.radar_proj(radar_bev) + self.radar_type_embed    # (B, D, H, W)
        pse_tok = self.pseudo_proj(pseudo_bev) + self.pseudo_type_embed  # (B, D, H, W)

        # Downsample to manageable resolution for attention
        # 350x350=122,500 tokens is too large; 32x32=1,024 tokens is feasible
        img_ds = F.adaptive_avg_pool2d(img_tok, (S, S))  # (B, D, S, S)
        rad_ds = F.adaptive_avg_pool2d(rad_tok, (S, S))
        pse_ds = F.adaptive_avg_pool2d(pse_tok, (S, S))

        N = S * S  # number of spatial tokens per modality after downsampling

        # Flatten to sequences: (B, N, D)
        img_seq = img_ds.flatten(2).permute(0, 2, 1)   # (B, N, D)
        rad_seq = rad_ds.flatten(2).permute(0, 2, 1)   # (B, N, D)
        pse_seq = pse_ds.flatten(2).permute(0, 2, 1)   # (B, N, D)

        # Concatenate all modality tokens: (B, 3*N, D)
        all_tokens = torch.cat([img_seq, rad_seq, pse_seq], dim=1)

        # Self-attention: every token attends to every other token
        # This enables: image↔radar, image↔pseudo, radar↔pseudo interactions
        attn_out, _ = self.self_attention(all_tokens, all_tokens, all_tokens)

        # Residual + LayerNorm + FFN
        attn_out = self.norm1(all_tokens + attn_out)
        attn_out = self.norm2(attn_out + self.ffn(attn_out))

        # Split back into per-modality outputs
        img_out = attn_out[:, 0:N, :]        # (B, N, D)
        rad_out = attn_out[:, N:2*N, :]      # (B, N, D)
        pse_out = attn_out[:, 2*N:3*N, :]    # (B, N, D)

        # Reshape each back to spatial (at downsampled resolution)
        img_spatial = img_out.permute(0, 2, 1).reshape(B, self.embed_dim, S, S)
        rad_spatial = rad_out.permute(0, 2, 1).reshape(B, self.embed_dim, S, S)
        pse_spatial = pse_out.permute(0, 2, 1).reshape(B, self.embed_dim, S, S)

        # Upsample back to original BEV resolution
        img_spatial = F.interpolate(img_spatial, size=(H, W), mode='bilinear', align_corners=False)
        rad_spatial = F.interpolate(rad_spatial, size=(H, W), mode='bilinear', align_corners=False)
        pse_spatial = F.interpolate(pse_spatial, size=(H, W), mode='bilinear', align_corners=False)

        # Merge all three enhanced modalities
        merged = torch.cat([img_spatial, rad_spatial, pse_spatial], dim=1)  # (B, 3D, H, W)
        fused = self.output_proj(merged)  # (B, D, H, W)

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
