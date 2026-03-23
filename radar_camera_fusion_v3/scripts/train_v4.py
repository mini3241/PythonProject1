"""
训练脚本v4 - 匹配论文超参数设置。
改进：
  1. 学习率 1e-3，Adam (β1=0.9, β2=0.999)
  2. StepLR 每5个epoch衰减0.7
  3. 训练73个epoch
  4. batch_size=4
  5. 检测损失权重提高
  6. Gaussian sigma=6 便于学习
  7. 可从v3 checkpoint恢复训练
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion_v3.config.base import BaseConfig
from radar_camera_fusion_v3.models.base_model import RadarCameraFusionModel
from radar_camera_fusion_v3.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion_v3.utils.tracker import Detection, FusionState
from radar_camera_fusion_v3.utils.metrics import compute_mota_motp, accumulate_mota_stats


class GaussianFocalLoss(nn.Module):
    """CenterNet风格的Focal Loss，处理BEV热图极端类别不平衡。"""

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        pred_sig = torch.clamp(pred_sig, min=1e-6, max=1 - 1e-6)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -torch.log(pred_sig) * torch.pow(1 - pred_sig, self.alpha) * pos_mask
        neg_loss = (-torch.log(1 - pred_sig) * torch.pow(pred_sig, self.alpha)
                    * torch.pow(1 - target, self.beta) * neg_mask)

        num_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class ScaleInvariantDepthLoss(nn.Module):
    """尺度不变深度损失（Eigen et al., 2014），论文公式1。"""

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


class UncertaintyDepthLoss(nn.Module):
    """
    不确定性感知深度损失（论文公式1完整版）。
    L_depth = Σ 1/2 exp(-s_{u,v}) |d - d^gt| + 1/2 s_{u,v}
    利用log_var（s_{u,v}）让网络学习在大误差区域预测高不确定性。
    """

    def forward(self, depth_pred, depth_gt, log_var):
        mask = (depth_gt > 0).detach()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=depth_pred.device, requires_grad=True)

        d = depth_pred[:, 0:1][mask]
        d_gt = depth_gt[mask]
        s = log_var[:, 0:1][mask]

        # 论文公式1
        loss = 0.5 * torch.exp(-s) * torch.abs(d - d_gt) + 0.5 * s
        return loss.mean()


def nms_heatmap(heatmap, kernel_size=3):
    """热图NMS：只保留局部最大值。"""
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


class TrainerV4:
    """训练管线v4 - 匹配论文设置。"""

    def __init__(self, config: BaseConfig, log_dir: str = './runs_v4',
                 resume_from: str = None):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # 日志设置
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'./logs/train_v4_{timestamp}.log'
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
        self.logger.info(f"训练日志: {log_file}")

        # 创建模型
        self.model = RadarCameraFusionModel(config).to(self.device)

        # 论文设置: Adam, lr=1e-3, β1=0.9, β2=0.999
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )

        # 从checkpoint恢复
        self.start_epoch = 1
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            # 尝试加载权重，忽略新增模块不匹配的key
            model_dict = self.model.state_dict()
            pretrained = {k: v for k, v in checkpoint['model_state_dict'].items()
                         if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            self.model.load_state_dict(model_dict)
            self.logger.info(f"从 {resume_from} 加载了 {len(pretrained)}/{len(model_dict)} 个参数")

        # 数据集
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
            batch_size=1,  # 验证用batch_size=1
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )

        # 损失函数
        self.detection_loss_fn = GaussianFocalLoss(alpha=2.0, beta=4.0)
        self.depth_loss_fn = ScaleInvariantDepthLoss(si_weight=0.5)
        self.uncertainty_depth_loss = UncertaintyDepthLoss()

        # 损失权重 - 提高检测损失权重
        self.alpha_detection = 5.0
        self.lambda_depth = 3.0
        self.lambda_uncertainty = 1.0

        # 高斯sigma - 较大的sigma便于学习
        self.gaussian_sigma = 6

        # 检测阈值
        self.det_threshold = 0.3

        self.writer = SummaryWriter(log_dir=log_dir)

        self.logger.info(f"=== 训练配置 V4 ===")
        self.logger.info(f"样本数: 训练={len(self.train_dataset)}, 验证={len(self.valid_dataset)}")
        self.logger.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"高斯sigma: {self.gaussian_sigma}")
        self.logger.info(f"损失权重: det={self.alpha_detection}, depth={self.lambda_depth}, unc={self.lambda_uncertainty}")
        self.logger.info(f"学习率: 1e-3, StepLR(step=5, gamma=0.7)")
        self.logger.info(f"检测阈值: {self.det_threshold}")

    def _create_target_heatmap(self, gt_positions):
        """创建目标热图（sigma=6的高斯）。"""
        if not isinstance(gt_positions, list):
            gt_positions = [gt_positions]

        batch_size = len(gt_positions)
        heatmap = torch.zeros(batch_size, 1, self.config.bev_height, self.config.bev_width)

        sigma = self.gaussian_sigma
        radius = sigma * 3

        for batch_idx, positions in enumerate(gt_positions):
            if len(positions) == 0:
                continue
            for pos in positions:
                x, y = pos[0].item(), pos[1].item()
                norm_x = (x - self.config.bev_x_range[0]) / (self.config.bev_x_range[1] - self.config.bev_x_range[0])
                norm_y = (y - self.config.bev_y_range[0]) / (self.config.bev_y_range[1] - self.config.bev_y_range[0])
                col_idx = int(norm_x * (self.config.bev_width - 1))
                row_idx = int(norm_y * (self.config.bev_height - 1))

                if 0 <= col_idx < self.config.bev_width and 0 <= row_idx < self.config.bev_height:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            c = col_idx + dx
                            r = row_idx + dy
                            if 0 <= c < self.config.bev_width and 0 <= r < self.config.bev_height:
                                dist_sq = dx * dx + dy * dy
                                value = np.exp(-dist_sq / (2 * sigma * sigma))
                                heatmap[batch_idx, 0, r, c] = max(heatmap[batch_idx, 0, r, c].item(), value)

        return heatmap

    def _get_gt_from_batch(self, batch):
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
        """训练一个epoch。"""
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

            # 计算损失
            detection_loss = torch.tensor(0.0, device=self.device)
            depth_loss = torch.tensor(0.0, device=self.device)
            unc_loss = torch.tensor(0.0, device=self.device)

            # 检测损失
            if 'detection_map' in outputs and batch['gt_positions'] is not None:
                target_heatmap = self._create_target_heatmap(batch['gt_positions'])
                target_heatmap = target_heatmap.to(self.device)
                detection_loss = self.detection_loss_fn(outputs['detection_map'], target_heatmap)

            # 深度损失 + 不确定性损失（论文公式1）
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

                # 不确定性损失
                log_var = outputs.get('log_var', None)
                if log_var is not None:
                    if log_var.shape[2:] != lidar_depth_gt.shape[2:]:
                        log_var_resized = F.interpolate(
                            log_var, size=lidar_depth_gt.shape[2:],
                            mode='bilinear', align_corners=False
                        )
                    else:
                        log_var_resized = log_var
                    unc_loss = self.uncertainty_depth_loss(depth_pred, lidar_depth_gt, log_var_resized)
                    unc_loss = torch.clamp(unc_loss, max=50.0)

            loss = (self.alpha_detection * detection_loss
                    + self.lambda_depth * depth_loss
                    + self.lambda_uncertainty * unc_loss)

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'det': f'{detection_loss.item():.4f}',
                    'dep': f'{depth_loss.item():.4f}',
                })

                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/total', loss.item(), global_step)
                self.writer.add_scalar('Loss/detection', detection_loss.item(), global_step)
                self.writer.add_scalar('Loss/depth', depth_loss.item(), global_step)

                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch} Batch {batch_idx}/{len(self.train_loader)} - "
                        f"Loss: {loss.item():.6f}, Det: {detection_loss.item():.6f}, "
                        f"Depth: {depth_loss.item():.6f}, Unc: {unc_loss.item():.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Epoch {epoch} - 平均损失: {avg_loss:.4f}")
        self.writer.add_scalar('Loss/epoch_avg', avg_loss, epoch)
        return avg_loss

    def _extract_detections(self, outputs, threshold=None):
        """从模型输出提取检测结果（含NMS）。"""
        if threshold is None:
            threshold = self.det_threshold
        detections = []

        if 'detection_map' not in outputs:
            return detections

        raw_heatmap = outputs['detection_map']
        heatmap_sig = torch.sigmoid(raw_heatmap)
        heatmap_nms = nms_heatmap(heatmap_sig, kernel_size=5)
        detection_map = heatmap_nms[0, 0].cpu().numpy()

        peaks = np.where(detection_map > threshold)

        for y, x in zip(peaks[0], peaks[1]):
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
        """验证 - 使用per-frame MOTA。"""
        self.model.eval()
        all_stats = []

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc=f"验证 {epoch}")
            for batch in pbar:
                gt_positions, gt_ids = self._get_gt_from_batch(batch)

                if len(gt_positions) == 0 or (gt_positions.ndim == 2 and gt_positions.shape[0] == 0):
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
                detections = self._extract_detections(outputs)

                if len(detections) > 0:
                    pred_positions = np.array([[d.center[0], d.center[1]] for d in detections])
                    pred_ids = np.arange(len(detections))
                else:
                    pred_positions = np.zeros((0, 2), dtype=np.float32)
                    pred_ids = np.zeros((0,), dtype=np.int32)

                mota, motp, stats = compute_mota_motp(
                    gt_positions, gt_ids, pred_positions, pred_ids
                )
                all_stats.append(stats)

        if all_stats:
            overall_mota, overall_motp = accumulate_mota_stats(all_stats)
            total_matches = sum(s['matches'] for s in all_stats)
            total_fp = sum(s['FP'] for s in all_stats)
            total_fn = sum(s['FN'] for s in all_stats)
            total_gt = sum(s['num_gt'] for s in all_stats)
            recall = total_matches / max(total_gt, 1)

            self.logger.info(
                f"验证 Epoch {epoch} - MOTA: {overall_mota:.4f}, MOTP: {overall_motp:.4f}, "
                f"Recall: {recall:.4f} ({total_matches}/{total_gt}), FP: {total_fp}, FN: {total_fn}")

            self.writer.add_scalar('Metrics/MOTA', overall_mota, epoch)
            self.writer.add_scalar('Metrics/MOTP', overall_motp, epoch)
            self.writer.add_scalar('Metrics/Recall', recall, epoch)
            self.writer.add_scalar('Metrics/FP', total_fp, epoch)
            self.writer.add_scalar('Metrics/FN', total_fn, epoch)

            return overall_mota
        else:
            self.logger.warning(f"验证 Epoch {epoch} - 无有效统计数据")
            return 0.0

    def save_checkpoint(self, epoch: int, save_path: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        self.logger.info(f"保存checkpoint: {save_path}")

    def train(self, num_epochs: int, save_dir: str):
        """主训练循环 - 论文设置：73 epoch, StepLR(step=5, gamma=0.7)。"""
        os.makedirs(save_dir, exist_ok=True)

        # 论文: StepLR 每5个epoch衰减0.7
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.7
        )

        best_mota = -999.0
        patience = 15
        no_improve = 0

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch} | LR: {lr:.6f}")
            self.logger.info(f"{'='*60}")

            train_loss = self.train_epoch(epoch)
            scheduler.step()

            # 保存checkpoint
            save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            self.save_checkpoint(epoch, save_path)
            latest_path = os.path.join(save_dir, 'latest_model.pth')
            self.save_checkpoint(epoch, latest_path)

            # 验证
            mota = self.validate(epoch)

            if mota > best_mota:
                best_mota = mota
                no_improve = 0
                save_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(epoch, save_path)
                self.logger.info(f"新最优模型! MOTA: {best_mota:.4f}")
            else:
                no_improve += 1
                self.logger.info(f"未改善 {no_improve}/{patience} epoch")

            if no_improve >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.logger.info(f"\n训练完成. 最优MOTA: {best_mota:.4f}")
        self.writer.close()


def main():
    config = BaseConfig(
        batch_size=4,
        learning_rate=1e-3,
    )
    trainer = TrainerV4(
        config,
        log_dir='./runs_v4',
        resume_from='./checkpoints_v3/best_model.pth'  # 可选：从v3恢复
    )
    # 论文: 训练73个epoch
    trainer.train(num_epochs=73, save_dir='./checkpoints_v4')


if __name__ == '__main__':
    main()
