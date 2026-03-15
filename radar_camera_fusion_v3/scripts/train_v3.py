"""
Training script v3 for radar-camera fusion model (v3 architecture).
Key fixes over train_v2.py:
  1. Fixed heatmap coordinate bug (X/Y were transposed in target generation)
  2. NMS peak extraction to reduce false positives
  3. Gaussian Focal Loss for better sparse heatmap learning
  4. Gradient clipping for stability
  5. Per-frame MOTA without tracker dependency
  6. FP/FN diagnostic logging
  7. Validate every epoch
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion_v3.config.base import BaseConfig
from radar_camera_fusion_v3.models.base_model import RadarCameraFusionModel
from radar_camera_fusion_v3.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion_v3.utils.tracker import Detection, FusionState
from radar_camera_fusion_v3.utils.metrics import compute_mota_motp, accumulate_mota_stats


class GaussianFocalLoss(nn.Module):
    """
    Focal loss for heatmap training (CenterNet style).
    Handles the severe class imbalance between positive and negative pixels.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha  # focusing parameter for negative samples
        self.beta = beta    # weight reduction for near-positive negatives

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) raw logits (before sigmoid)
            target: (B, 1, H, W) Gaussian heatmap target [0, 1]
        """
        pred_sig = torch.sigmoid(pred)
        pred_sig = torch.clamp(pred_sig, min=1e-6, max=1 - 1e-6)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive loss: -log(p) * (1-p)^alpha
        pos_loss = -torch.log(pred_sig) * torch.pow(1 - pred_sig, self.alpha) * pos_mask

        # Negative loss: -log(1-p) * p^alpha * (1-gt)^beta
        neg_loss = (-torch.log(1 - pred_sig) * torch.pow(pred_sig, self.alpha)
                    * torch.pow(1 - target, self.beta) * neg_mask)

        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class ScaleInvariantDepthLoss(nn.Module):
    """Scale-Invariant depth loss (Eigen et al., 2014)."""

    def __init__(self, si_weight: float = 0.5):
        super().__init__()
        self.si_weight = si_weight

    def forward(self, prediction, gt):
        if torch.isnan(prediction).any():
            prediction = torch.nan_to_num(prediction, nan=0.0)
        if torch.isnan(gt).any():
            gt = torch.nan_to_num(gt, nan=0.0)

        prediction = prediction[:, 0:1]
        mask = (gt > 0).detach()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        pred_valid = prediction[mask]
        gt_valid = gt[mask]

        diff = torch.log(pred_valid.clamp(min=0.1)) - torch.log(gt_valid.clamp(min=0.1))
        loss = torch.mean(diff ** 2) - self.si_weight * (torch.mean(diff) ** 2)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        return loss


class SmoothEdgeLoss(nn.Module):
    """Smooth edge loss for depth estimation."""

    def __init__(self):
        super().__init__()
        self.alpha = 0.2
        self.beta = 1.2

    def forward(self, depth, img):
        if torch.isnan(depth).any():
            depth = torch.nan_to_num(depth, nan=0.0)
        if torch.isnan(img).any():
            img = torch.nan_to_num(img, nan=0.0)

        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        grad_seg_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_seg_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_seg_x_range = torch.max(grad_seg_x) - torch.min(grad_seg_x)
        if grad_seg_x_range > 1e-8:
            grad_seg_x = (grad_seg_x - torch.min(grad_seg_x)) / grad_seg_x_range

        grad_seg_y_range = torch.max(grad_seg_y) - torch.min(grad_seg_y)
        if grad_seg_y_range > 1e-8:
            grad_seg_y = (grad_seg_y - torch.min(grad_seg_y)) / grad_seg_y_range

        grad_seg_x = torch.pow(grad_seg_x, self.alpha)
        grad_seg_y = torch.pow(grad_seg_y, self.alpha)

        loss_x = grad_depth_x * torch.exp(-self.beta * grad_seg_x)
        loss_y = grad_depth_y * torch.exp(-self.beta * grad_seg_y)

        return torch.mean(loss_x) + torch.mean(loss_y)


def nms_heatmap(heatmap, kernel_size=3):
    """Simple NMS on heatmap: keep only local maxima."""
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


class TrainerV3:
    """Training pipeline v3 with coordinate fix and improved detection."""

    def __init__(self, config: BaseConfig, log_dir: str = './runs'):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'./logs/train_v3_{timestamp}.log'
        os.makedirs('./logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training log: {log_file}")

        # Create model
        self.model = RadarCameraFusionModel(config).to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create datasets
        train_list = os.path.join(os.path.dirname(config.mapping_csv), 'train.txt')
        valid_list = os.path.join(os.path.dirname(config.mapping_csv), 'valid.txt')

        self.train_dataset = RadarCameraDataset(config, train_list, is_train=True)
        self.valid_dataset = RadarCameraDataset(config, valid_list, is_train=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )

        # Loss functions
        self.detection_loss_fn = GaussianFocalLoss(alpha=2.0, beta=4.0)
        self.depth_loss_fn = ScaleInvariantDepthLoss(si_weight=0.5)
        self.smooth_edge_loss = SmoothEdgeLoss()

        # Loss weights
        self.alpha_detection = 1.0
        self.lambda_depth = 5.0
        self.lambda_smooth = 0.1

        # Detection threshold for validation
        self.det_threshold = 0.3

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        self.logger.info(f"Initialized trainer with {len(self.train_dataset)} training samples")
        self.logger.info(f"Validation set: {len(self.valid_dataset)} samples")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"TensorBoard logs: {log_dir}")
        self.logger.info(f"Detection loss: GaussianFocalLoss (alpha=2.0, beta=4.0)")
        self.logger.info(f"Detection threshold: {self.det_threshold}")
        self.logger.info(f"BEV grid: {config.bev_height} x {config.bev_width} "
                        f"(x: {config.bev_x_range}, y: {config.bev_y_range})")
        self.logger.info(f"Config: batch_size={config.batch_size}, lr={config.learning_rate}, "
                        f"bev_resolution={config.bev_resolution}, image_size={config.image_size}")

    def _create_target_heatmap(self, gt_positions):
        """
        Create target heatmap from ground truth positions with Gaussian.

        COORDINATE CONVENTION (fixed in v3):
          heatmap shape: (B, 1, H, W) where H=bev_height (Y-axis), W=bev_width (X-axis)
          heatmap[b, 0, row, col] where row=Y index, col=X index
        """
        if not isinstance(gt_positions, list):
            gt_positions = [gt_positions]

        batch_size = len(gt_positions)
        heatmap = torch.zeros(
            batch_size, 1,
            self.config.bev_height,
            self.config.bev_width
        )

        sigma = 3  # Gaussian standard deviation in pixels

        for batch_idx, positions in enumerate(gt_positions):
            if len(positions) == 0:
                continue

            for pos in positions:
                x, y = pos[0].item(), pos[1].item()

                # Convert world coordinates to grid indices
                # col_idx = X index (0 to bev_width-1)
                # row_idx = Y index (0 to bev_height-1)
                norm_x = (x - self.config.bev_x_range[0]) / (self.config.bev_x_range[1] - self.config.bev_x_range[0])
                norm_y = (y - self.config.bev_y_range[0]) / (self.config.bev_y_range[1] - self.config.bev_y_range[0])

                col_idx = int(norm_x * (self.config.bev_width - 1))
                row_idx = int(norm_y * (self.config.bev_height - 1))

                if 0 <= col_idx < self.config.bev_width and 0 <= row_idx < self.config.bev_height:
                    radius = sigma * 3
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            c = col_idx + dx
                            r = row_idx + dy
                            if 0 <= c < self.config.bev_width and 0 <= r < self.config.bev_height:
                                dist_sq = dx * dx + dy * dy
                                value = np.exp(-dist_sq / (2 * sigma * sigma))
                                # heatmap[b, 0, row, col] — row=Y, col=X
                                heatmap[batch_idx, 0, r, c] = max(heatmap[batch_idx, 0, r, c].item(), value)

        return heatmap

    def _get_gt_from_batch(self, batch):
        """Extract GT positions and IDs, handling both batch_size=1 and >1."""
        gt_pos_raw = batch['gt_positions']
        gt_ids_raw = batch['gt_ids']

        if isinstance(gt_pos_raw, list):
            gt_positions = gt_pos_raw[0].cpu().numpy()
            gt_ids = gt_ids_raw[0].cpu().numpy()
        else:
            gt_positions = gt_pos_raw.cpu().numpy()
            gt_ids = gt_ids_raw.cpu().numpy()

        return gt_positions, gt_ids

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            gt_positions = batch['gt_positions']
            if isinstance(gt_positions, list):
                if len(gt_positions) == 0 or gt_positions[0].size(0) == 0:
                    continue
            elif gt_positions.size(0) == 0:
                continue

            images = batch['images'].to(self.device)
            radar_points = batch['radar_points']
            if isinstance(radar_points, list):
                radar_points = [rp.to(self.device) for rp in radar_points]
            else:
                radar_points = radar_points.to(self.device)

            intrinsic_matrix = batch.get('intrinsic_matrix')
            if intrinsic_matrix is not None:
                intrinsic_matrix = intrinsic_matrix.to(self.device)
            lidar_to_camera_extrinsic = batch.get('lidar_to_camera_extrinsic')
            if lidar_to_camera_extrinsic is not None:
                lidar_to_camera_extrinsic = lidar_to_camera_extrinsic.to(self.device)

            self.optimizer.zero_grad()

            model_input = {
                'images': images,
                'radar_points': radar_points
            }
            if intrinsic_matrix is not None:
                model_input['intrinsic_matrix'] = intrinsic_matrix
            if lidar_to_camera_extrinsic is not None:
                model_input['lidar_to_camera_extrinsic'] = lidar_to_camera_extrinsic

            outputs = self.model(model_input)

            # Compute losses
            detection_loss = torch.tensor(0.0, device=self.device)
            depth_loss = torch.tensor(0.0, device=self.device)
            smooth_loss = torch.tensor(0.0, device=self.device)

            # Detection loss
            if 'detection_map' in outputs and batch['gt_positions'] is not None:
                target_heatmap = self._create_target_heatmap(batch['gt_positions'])
                target_heatmap = target_heatmap.to(self.device)
                detection_loss = self.detection_loss_fn(outputs['detection_map'], target_heatmap)

            # Depth loss
            if 'depth_map' in outputs and 'lidar_depth' in batch:
                depth_pred = outputs['depth_map']
                lidar_depth_gt = batch['lidar_depth'].to(self.device)

                if lidar_depth_gt.dim() == 3:
                    lidar_depth_gt = lidar_depth_gt.unsqueeze(1)

                if depth_pred.shape[2:] != lidar_depth_gt.shape[2:]:
                    lidar_depth_gt = F.interpolate(
                        lidar_depth_gt, size=depth_pred.shape[2:],
                        mode='bilinear', align_corners=False
                    )

                depth_loss = self.depth_loss_fn(depth_pred, lidar_depth_gt)
                depth_loss = torch.clamp(depth_loss, max=100.0)

                if images is not None:
                    if depth_pred.shape[2:] != images.shape[2:]:
                        depth_pred_resized = F.interpolate(
                            depth_pred, size=images.shape[2:],
                            mode='bilinear', align_corners=False
                        )
                    else:
                        depth_pred_resized = depth_pred
                    smooth_loss = self.smooth_edge_loss(depth_pred_resized, images)
                    smooth_loss = torch.clamp(smooth_loss, max=10.0)

            loss = (self.alpha_detection * detection_loss
                    + self.lambda_depth * depth_loss
                    + self.lambda_smooth * smooth_loss)

            if loss > 0:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'total': f'{loss.item():.4f}',
                    'det': f'{detection_loss.item():.4f}',
                    'depth': f'{depth_loss.item():.4f}',
                    'smooth': f'{smooth_loss.item():.4f}'
                })

                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_total', loss.item(), global_step)
                self.writer.add_scalar('Loss/train_detection', detection_loss.item(), global_step)
                self.writer.add_scalar('Loss/train_depth', depth_loss.item(), global_step)
                self.writer.add_scalar('Loss/train_smooth', smooth_loss.item(), global_step)

                if batch_idx % 50 == 0:
                    self.logger.info(
                        f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} - "
                        f"Total: {loss.item():.6f}, Det: {detection_loss.item():.6f}, "
                        f"Depth: {depth_loss.item():.6f}, Smooth: {smooth_loss.item():.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        self.writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        return avg_loss

    def _extract_detections(self, outputs, threshold=None):
        """Extract detections with NMS from model outputs."""
        if threshold is None:
            threshold = self.det_threshold

        detections = []

        if 'detection_map' not in outputs:
            return detections

        # Sigmoid + NMS
        raw_heatmap = outputs['detection_map']  # (B, 1, H, W)
        heatmap_sig = torch.sigmoid(raw_heatmap)
        heatmap_nms = nms_heatmap(heatmap_sig, kernel_size=5)
        detection_map = heatmap_nms[0, 0].cpu().numpy()

        peaks = np.where(detection_map > threshold)

        for y, x in zip(peaks[0], peaks[1]):
            # row=y → Y world coordinate, col=x → X world coordinate
            world_x = (x / self.config.bev_width
                       * (self.config.bev_x_range[1] - self.config.bev_x_range[0])
                       + self.config.bev_x_range[0])
            world_y = (y / self.config.bev_height
                       * (self.config.bev_y_range[1] - self.config.bev_y_range[0])
                       + self.config.bev_y_range[0])

            detections.append(Detection(
                center=(world_x, world_y),
                confidence=float(detection_map[y, x]),
                fusion_state=FusionState.FUSED
            ))

        return detections

    def validate(self, epoch: int):
        """Validate the model with per-frame MOTA."""
        self.model.eval()
        all_stats = []

        diag_total = 0
        diag_skipped = 0
        diag_no_det_map = 0
        diag_no_det = 0
        diag_has_det = 0
        diag_total_fp = 0
        diag_total_fn = 0
        diag_total_gt = 0

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f"Validation {epoch}")
            for batch in pbar:
                diag_total += 1

                gt_positions, gt_ids = self._get_gt_from_batch(batch)

                if len(gt_positions) == 0 or (gt_positions.ndim == 2 and gt_positions.shape[0] == 0):
                    diag_skipped += 1
                    continue

                if gt_positions.ndim == 1:
                    gt_positions = gt_positions.reshape(1, -1)

                images = batch['images'].to(self.device)
                radar_points = batch['radar_points']
                if isinstance(radar_points, list):
                    radar_points = [rp.to(self.device) for rp in radar_points]
                else:
                    radar_points = radar_points.to(self.device)
                intrinsic_matrix = batch.get('intrinsic_matrix')
                if intrinsic_matrix is not None:
                    intrinsic_matrix = intrinsic_matrix.to(self.device)
                lidar_to_camera_extrinsic = batch.get('lidar_to_camera_extrinsic')
                if lidar_to_camera_extrinsic is not None:
                    lidar_to_camera_extrinsic = lidar_to_camera_extrinsic.to(self.device)

                model_input = {
                    'images': images,
                    'radar_points': radar_points
                }
                if intrinsic_matrix is not None:
                    model_input['intrinsic_matrix'] = intrinsic_matrix
                if lidar_to_camera_extrinsic is not None:
                    model_input['lidar_to_camera_extrinsic'] = lidar_to_camera_extrinsic

                outputs = self.model(model_input)

                # Diagnostic on first batch
                if diag_total == 1:
                    self.logger.info(f"[DIAG] Model output keys: {list(outputs.keys())}")
                    if 'detection_map' in outputs:
                        det_map = torch.sigmoid(outputs['detection_map'][0, 0])
                        self.logger.info(
                            f"[DIAG] detection_map shape: {det_map.shape}, "
                            f"min: {det_map.min().item():.4f}, "
                            f"max: {det_map.max().item():.4f}, "
                            f"mean: {det_map.mean().item():.4f}")

                detections = self._extract_detections(outputs)

                if 'detection_map' not in outputs:
                    diag_no_det_map += 1

                if len(detections) == 0:
                    diag_no_det += 1
                else:
                    diag_has_det += 1

                # Per-frame MOTA
                if len(detections) > 0:
                    pred_positions = np.array([[d.center[0], d.center[1]] for d in detections])
                    pred_ids = np.arange(len(detections))
                else:
                    pred_positions = np.zeros((0, 2), dtype=np.float32)
                    pred_ids = np.zeros((0,), dtype=np.int32)

                mota, motp, stats = compute_mota_motp(
                    gt_positions, gt_ids,
                    pred_positions, pred_ids
                )
                all_stats.append(stats)

                diag_total_fp += stats['FP']
                diag_total_fn += stats['FN']
                diag_total_gt += stats['num_gt']

        # Print summary
        self.logger.info(f"[DIAG] Validation Epoch {epoch} summary:")
        self.logger.info(f"  Total batches: {diag_total}, Skipped (empty GT): {diag_skipped}")
        self.logger.info(f"  No detection_map: {diag_no_det_map}")
        self.logger.info(f"  Batches with detections: {diag_has_det}, without: {diag_no_det}")
        self.logger.info(f"  Total FP: {diag_total_fp}, Total FN: {diag_total_fn}, Total GT: {diag_total_gt}")

        if all_stats:
            overall_mota, overall_motp = accumulate_mota_stats(all_stats)
            self.logger.info(f"Validation Epoch {epoch} - MOTA: {overall_mota:.4f}, MOTP: {overall_motp:.4f}")

            self.writer.add_scalar('Metrics/MOTA', overall_mota, epoch)
            self.writer.add_scalar('Metrics/MOTP', overall_motp, epoch)
            self.writer.add_scalar('Metrics/FP', diag_total_fp, epoch)
            self.writer.add_scalar('Metrics/FN', diag_total_fn, epoch)

            return overall_mota
        else:
            self.logger.warning(f"Validation Epoch {epoch} - No valid statistics")
            return 0.0

    def save_checkpoint(self, epoch: int, save_path: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

    def train(self, num_epochs: int, save_dir: str):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)

        best_mota = -999.0

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{num_epochs}")
            self.logger.info(f"{'='*60}")

            train_loss = self.train_epoch(epoch)

            save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            self.save_checkpoint(epoch, save_path)

            latest_path = os.path.join(save_dir, 'latest_model.pth')
            self.save_checkpoint(epoch, latest_path)

            # Validate every epoch
            mota = self.validate(epoch)

            if mota > best_mota:
                best_mota = mota
                save_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(epoch, save_path)
                self.logger.info(f"New best model! MOTA: {best_mota:.4f}")

        self.logger.info(f"\nTraining completed. Best MOTA: {best_mota:.4f}")
        self.writer.close()


def main():
    config = BaseConfig()
    trainer = TrainerV3(config)
    trainer.train(num_epochs=20, save_dir='./checkpoints_v3')


if __name__ == '__main__':
    main()
