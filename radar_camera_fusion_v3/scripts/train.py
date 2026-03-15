"""
Training script for radar-camera fusion model.
Clean, modular implementation.
"""

import os
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

# Add parent directory to path
sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion_v3.config.base import BaseConfig
from radar_camera_fusion_v3.models.base_model import RadarCameraFusionModel
from radar_camera_fusion_v3.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion_v3.utils.tracker import SequenceMOTATracker, Detection, FusionState
from radar_camera_fusion_v3.utils.metrics import compute_mota_motp, accumulate_mota_stats
import torch.nn.functional as F


class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-Invariant depth loss (Eigen et al., 2014).
    Better suited for sparse GT than plain MAE: penalizes relative depth errors
    and adds a variance term to discourage trivial constant-depth predictions.
    """

    def __init__(self, si_weight: float = 0.5):
        super().__init__()
        self.si_weight = si_weight  # weight for the scale-invariant (variance) term

    def forward(self, prediction, gt):
        """
        Args:
            prediction: (B, 1, H, W) depth prediction in meters
            gt: (B, 1, H, W) depth ground truth (0 = invalid)
        """
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

        # log-space difference (clamp to avoid log(0))
        diff = torch.log(pred_valid.clamp(min=0.1)) - torch.log(gt_valid.clamp(min=0.1))

        # MSE in log space + scale-invariant variance penalty
        loss = torch.mean(diff ** 2) - self.si_weight * (torch.mean(diff) ** 2)

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        return loss


class SmoothEdgeLoss(nn.Module):
    """Smooth edge loss for depth estimation aligned with SGDNet_TI."""

    def __init__(self):
        super(SmoothEdgeLoss, self).__init__()
        self.alpha = 0.2
        self.beta = 1.2

    def forward(self, depth, img):
        """
        Args:
            depth: (B, 1, H, W) depth prediction
            img: (B, 3, H, W) image (for edge computation)
        """
        # Handle NaN
        if torch.isnan(depth).any():
            depth = torch.nan_to_num(depth, nan=0.0)

        if torch.isnan(img).any():
            img = torch.nan_to_num(img, nan=0.0)

        # Compute gradients
        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        grad_seg_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_seg_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        # Normalize gradients
        grad_seg_x_min = torch.min(grad_seg_x)
        grad_seg_x_max = torch.max(grad_seg_x)
        if (grad_seg_x_max - grad_seg_x_min) > 1e-8:
            grad_seg_x = (grad_seg_x - grad_seg_x_min) / (grad_seg_x_max - grad_seg_x_min)

        grad_seg_y_min = torch.min(grad_seg_y)
        grad_seg_y_max = torch.max(grad_seg_y)
        if (grad_seg_y_max - grad_seg_y_min) > 1e-8:
            grad_seg_y = (grad_seg_y - grad_seg_y_min) / (grad_seg_y_max - grad_seg_y_min)

        # Compute smooth edge loss
        grad_seg_x = torch.pow(grad_seg_x, self.alpha)
        grad_seg_y = torch.pow(grad_seg_y, self.alpha)

        loss_x = grad_depth_x * torch.exp(-self.beta * grad_seg_x)
        loss_y = grad_depth_y * torch.exp(-self.beta * grad_seg_y)

        smooth_loss = torch.mean(loss_x) + torch.mean(loss_y)

        return smooth_loss


class Trainer:
    """Training pipeline for radar-camera fusion."""

    def __init__(self, config: BaseConfig, log_dir: str = './runs'):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'./logs/train_{timestamp}.log'
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
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.depth_loss_fn = ScaleInvariantDepthLoss(si_weight=0.5)
        self.smooth_edge_loss = SmoothEdgeLoss()

        # Loss weights — depth needs a strong signal to learn from sparse GT
        self.alpha_detection = 5.0
        self.lambda_depth = 5.0
        self.lambda_smooth = 0.1

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Tracker for evaluation
        self.tracker = SequenceMOTATracker()

        self.logger.info(f"Initialized trainer with {len(self.train_dataset)} training samples")
        self.logger.info(f"Validation set: {len(self.valid_dataset)} samples")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"TensorBoard logs: {log_dir}")
        self.logger.info(f"Config: batch_size={config.batch_size}, lr={config.learning_rate}, "
                        f"bev_resolution={config.bev_resolution}, image_size={config.image_size}")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Skip batches with empty ground truth
            gt_positions = batch['gt_positions']
            if isinstance(gt_positions, list):
                if len(gt_positions) == 0 or gt_positions[0].size(0) == 0:
                    continue
            elif gt_positions.size(0) == 0:
                continue

            # Move data to device
            images = batch['images'].to(self.device)

            # Handle radar_points - can be a tensor (batch_size=1) or list (batch_size>1)
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

            # Forward pass
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

            # Detection loss (if detection head exists)
            if 'detection_map' in outputs and batch['gt_positions'] is not None:
                # Create target heatmap from GT positions
                target_heatmap = self._create_target_heatmap(batch['gt_positions'])
                target_heatmap = target_heatmap.to(self.device)

                detection_loss = self.detection_loss_fn(
                    outputs['detection_map'],
                    target_heatmap
                )

            # Depth loss (if depth map is predicted)
            if 'depth_map' in outputs and 'lidar_depth' in batch:
                depth_pred = outputs['depth_map']
                lidar_depth_gt = batch['lidar_depth'].to(self.device)

                # Ensure GT has correct shape (B, H, W) -> (B, 1, H, W)
                if lidar_depth_gt.dim() == 3:
                    lidar_depth_gt = lidar_depth_gt.unsqueeze(1)

                # Resize GT to match depth prediction size if needed
                if depth_pred.shape[2:] != lidar_depth_gt.shape[2:]:
                    lidar_depth_gt = F.interpolate(
                        lidar_depth_gt,
                        size=depth_pred.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # MAE depth loss with ground truth
                depth_loss = self.depth_loss_fn(depth_pred, lidar_depth_gt)
                depth_loss = torch.clamp(depth_loss, max=100.0)

                # Smooth edge loss
                if images is not None:
                    # Resize depth to match image size if needed
                    if depth_pred.shape[2:] != images.shape[2:]:
                        depth_pred_resized = F.interpolate(
                            depth_pred,
                            size=images.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        depth_pred_resized = depth_pred

                    smooth_loss = self.smooth_edge_loss(depth_pred_resized, images)
                    smooth_loss = torch.clamp(smooth_loss, max=10.0)

            # Total loss
            loss = self.alpha_detection * detection_loss + self.lambda_depth * depth_loss + self.lambda_smooth * smooth_loss

            # Backward pass
            if loss > 0:
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Print detailed loss information
                loss_info = {
                    'total': f'{loss.item():.4f}',
                    'det': f'{detection_loss.item():.4f}',
                    'depth': f'{depth_loss.item():.4f}',
                    'smooth': f'{smooth_loss.item():.4f}'
                }
                pbar.set_postfix(loss_info)

                # Log to TensorBoard
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_total', loss.item(), global_step)
                self.writer.add_scalar('Loss/train_detection', detection_loss.item(), global_step)
                self.writer.add_scalar('Loss/train_depth', depth_loss.item(), global_step)
                self.writer.add_scalar('Loss/train_smooth', smooth_loss.item(), global_step)

                # Log detailed loss every 50 batches
                if batch_idx % 50 == 0:
                    self.logger.info(f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} - "
                                   f"Total: {loss.item():.6f}, Det: {detection_loss.item():.6f}, "
                                   f"Depth: {depth_loss.item():.6f}, Smooth: {smooth_loss.item():.6f}")

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # Log epoch average loss
        self.writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)

        return avg_loss

    def _create_target_heatmap(self, gt_positions):
        """Create target heatmap from ground truth positions with Gaussian."""
        # Handle both single sample and batch
        if not isinstance(gt_positions, list):
            gt_positions = [gt_positions]

        batch_size = len(gt_positions)
        heatmap = torch.zeros(
            batch_size, 1,
            self.config.bev_height,
            self.config.bev_width
        )

        # Gaussian kernel parameters
        sigma = 3  # Standard deviation in pixels

        for batch_idx, positions in enumerate(gt_positions):
            if len(positions) == 0:
                continue

            for pos in positions:
                x, y = pos[0].item(), pos[1].item()

                # Convert to grid coordinates
                norm_x = (x - self.config.bev_x_range[0]) / (self.config.bev_x_range[1] - self.config.bev_x_range[0])
                norm_y = (y - self.config.bev_y_range[0]) / (self.config.bev_y_range[1] - self.config.bev_y_range[0])

                grid_x = int(norm_x * (self.config.bev_width - 1))
                grid_y = int(norm_y * (self.config.bev_height - 1))

                if 0 <= grid_x < self.config.bev_width and 0 <= grid_y < self.config.bev_height:
                    # Apply Gaussian kernel around GT position
                    radius = sigma * 3  # 3-sigma rule
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            gx = grid_x + dx
                            gy = grid_y + dy
                            if 0 <= gx < self.config.bev_width and 0 <= gy < self.config.bev_height:
                                # Gaussian formula
                                dist_sq = dx * dx + dy * dy
                                value = torch.exp(torch.tensor(-dist_sq / (2 * sigma * sigma)))
                                # Take maximum to handle overlapping Gaussians
                                # IMPORTANT: Use [gx, gy] to match model's [X, Y] coordinate convention
                                heatmap[batch_idx, 0, gx, gy] = torch.max(heatmap[batch_idx, 0, gx, gy], value)

        return heatmap

    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        all_stats = []

        # Reset tracker
        self.tracker = SequenceMOTATracker()

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f"Validation {epoch}")
            for batch in pbar:
                # Skip batches with empty ground truth
                if len(batch['gt_positions']) == 0 or batch['gt_positions'][0].size(0) == 0:
                    continue

                images = batch['images'].to(self.device)
                radar_points = batch['radar_points']
                if isinstance(radar_points, list):
                    radar_points = [rp.to(self.device) for rp in radar_points]
                else:
                    radar_points = radar_points.to(self.device)
                gt_positions = batch['gt_positions'][0].cpu().numpy()
                gt_ids = batch['gt_ids'][0].cpu().numpy()
                intrinsic_matrix = batch.get('intrinsic_matrix')
                if intrinsic_matrix is not None:
                    intrinsic_matrix = intrinsic_matrix.to(self.device)
                lidar_to_camera_extrinsic = batch.get('lidar_to_camera_extrinsic')
                if lidar_to_camera_extrinsic is not None:
                    lidar_to_camera_extrinsic = lidar_to_camera_extrinsic.to(self.device)

                # Forward pass
                model_input = {
                    'images': images,
                    'radar_points': radar_points
                }
                if intrinsic_matrix is not None:
                    model_input['intrinsic_matrix'] = intrinsic_matrix
                if lidar_to_camera_extrinsic is not None:
                    model_input['lidar_to_camera_extrinsic'] = lidar_to_camera_extrinsic

                outputs = self.model(model_input)

                # Extract detections from output
                detections = self._extract_detections(outputs)

                # Update tracker
                self.tracker.update(detections)

                # Get tracked positions
                tracks = self.tracker.get_confirmed_tracks()
                pred_positions = np.array([t.position for t in tracks])
                pred_ids = np.array([t.track_id for t in tracks])

                # Compute MOTA/MOTP
                if len(pred_positions) > 0:
                    mota, motp, stats = compute_mota_motp(
                        gt_positions, gt_ids,
                        pred_positions, pred_ids
                    )
                    all_stats.append(stats)

        # Compute overall metrics
        if all_stats:
            overall_mota, overall_motp = accumulate_mota_stats(all_stats)
            self.logger.info(f"Validation Epoch {epoch} - MOTA: {overall_mota:.4f}, MOTP: {overall_motp:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar('Metrics/MOTA', overall_mota, epoch)
            self.writer.add_scalar('Metrics/MOTP', overall_motp, epoch)

            return overall_mota
        else:
            self.logger.warning(f"Validation Epoch {epoch} - No valid statistics")
            return 0.0

    def _extract_detections(self, outputs):
        """Extract detections from model outputs."""
        detections = []

        if 'detection_map' in outputs:
            detection_map = torch.sigmoid(outputs['detection_map'][0, 0]).cpu().numpy()

            # Simple peak detection
            threshold = 0.5
            peaks = np.where(detection_map > threshold)

            for y, x in zip(peaks[0], peaks[1]):
                # Convert grid to world coordinates
                world_x = (x / self.config.bev_width *
                          (self.config.bev_x_range[1] - self.config.bev_x_range[0]) +
                          self.config.bev_x_range[0])
                world_y = (y / self.config.bev_height *
                          (self.config.bev_y_range[1] - self.config.bev_y_range[0]) +
                          self.config.bev_y_range[0])

                detections.append(Detection(
                    center=(world_x, world_y),
                    confidence=detection_map[y, x],
                    fusion_state=FusionState.FUSED
                ))

        return detections

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

        best_mota = 0.0

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{num_epochs}")
            self.logger.info(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Save checkpoint after every epoch
            save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            self.save_checkpoint(epoch, save_path)

            # Also save as latest checkpoint
            latest_path = os.path.join(save_dir, 'latest_model.pth')
            self.save_checkpoint(epoch, latest_path)

            # Validate every 5 epochs
            if epoch % 5 == 0:
                mota = self.validate(epoch)

                # Save best model
                if mota > best_mota:
                    best_mota = mota
                    save_path = os.path.join(save_dir, f'best_model.pth')
                    self.save_checkpoint(epoch, save_path)
                    self.logger.info(f"New best model! MOTA: {best_mota:.4f}")

        self.logger.info(f"\nTraining completed. Best MOTA: {best_mota:.4f}")

        # Close TensorBoard writer
        self.writer.close()


def main():
    # Configuration
    config = BaseConfig()

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(
        num_epochs=20,
        save_dir='./checkpoints'
    )


if __name__ == '__main__':
    main()
