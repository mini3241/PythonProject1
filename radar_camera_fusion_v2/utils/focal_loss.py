"""
Gaussian Focal Loss for heatmap-based detection (CenterNet-style).
Handles extreme positive/negative imbalance in BEV heatmaps.

Usage:
    Replace BCEWithLogitsLoss in train.py:
        # from:
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        # to:
        from radar_camera_fusion_v2.utils.focal_loss import GaussianFocalLoss
        self.detection_loss_fn = GaussianFocalLoss()
"""

import torch
import torch.nn as nn


class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmap prediction (CenterNet-style).

    Standard BCEWithLogitsLoss treats all pixels equally, but BEV heatmaps
    are extremely sparse (~99.99% background). This causes the model to
    learn "predict all zeros" since it minimizes loss trivially.

    Gaussian Focal Loss (Law & Deng, CornerNet; Zhou et al., CenterNet):
    - Positive pixels (GT peaks): penalize low-confidence predictions harder
    - Negative pixels: suppress easy negatives near GT using (1-target)^beta
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha  # focusing parameter for hard negatives
        self.beta = beta    # suppression weight for easy negatives near GT

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) raw logits (before sigmoid)
            target: (B, 1, H, W) Gaussian heatmap in [0, 1]
        Returns:
            Scalar loss normalized by number of positive pixels
        """
        pred_sigmoid = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)

        # Positive locations (GT peak == 1.0)
        pos_mask = target.ge(1.0)
        # Negative locations (background < 1.0)
        neg_mask = target.lt(1.0)

        # Positive loss: -((1 - p)^alpha) * log(p)
        pos_loss = -((1 - pred_sigmoid) ** self.alpha) * torch.log(pred_sigmoid)
        pos_loss = pos_loss[pos_mask].sum()

        # Negative loss: -((1 - target)^beta) * (p^alpha) * log(1 - p)
        # (1 - target)^beta: pixels closer to GT center get smaller weight
        neg_loss = -((1 - target) ** self.beta) * (pred_sigmoid ** self.alpha) * torch.log(1 - pred_sigmoid)
        neg_loss = neg_loss[neg_mask].sum()

        num_pos = pos_mask.float().sum().clamp(min=1.0)
        loss = (pos_loss + neg_loss) / num_pos

        return loss
