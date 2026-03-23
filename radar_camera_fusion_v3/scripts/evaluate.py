"""
评估脚本（v3架构）- 支持时序累积TCA和自适应卡尔曼跟踪。
输出 per-frame MOTA 和 tracker-based MOTA。
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')


def nms_heatmap(heatmap, kernel_size=3):
    """热图NMS：只保留局部最大值。"""
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep

from radar_camera_fusion_v3.config.base import BaseConfig
from radar_camera_fusion_v3.models.base_model import RadarCameraFusionModel
from radar_camera_fusion_v3.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion_v3.utils.tracker import SequenceMOTATracker, Detection, FusionState
from radar_camera_fusion_v3.utils.metrics import compute_mota_motp, accumulate_mota_stats


class Evaluator:
    """评估管线 - 支持所有论文模块。"""

    def __init__(self, config: BaseConfig, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        self.model = RadarCameraFusionModel(config).to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # 灵活加载（兼容新旧模型）
            model_dict = self.model.state_dict()
            pretrained = {k: v for k, v in checkpoint['model_state_dict'].items()
                         if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            self.model.load_state_dict(model_dict)
            print(f"加载checkpoint: {checkpoint_path} ({len(pretrained)}/{len(model_dict)} 参数)")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
        else:
            print(f"警告: checkpoint未找到 {checkpoint_path}")

        self.model.eval()

        valid_list = os.path.join(os.path.dirname(config.mapping_csv), 'valid.txt')
        self.valid_dataset = RadarCameraDataset(config, valid_list, is_train=False)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=1, shuffle=False,
            num_workers=config.num_workers, collate_fn=custom_collate_fn
        )

        self.det_threshold = 0.3
        print(f"评估器初始化: {len(self.valid_dataset)} 验证样本, 检测阈值: {self.det_threshold}")

    def _extract_detections(self, outputs):
        """提取检测结果（含置信度，用于自适应卡尔曼噪声）。"""
        detections = []

        if 'detection_map' not in outputs:
            return detections

        raw_heatmap = outputs['detection_map']
        heatmap_sig = torch.sigmoid(raw_heatmap)
        heatmap_nms = nms_heatmap(heatmap_sig, kernel_size=5)
        detection_map = heatmap_nms[0, 0].cpu().numpy()

        # 获取融合置信度图（如果有TCA累积的置信度）
        fused_conf = outputs.get('fused_confidence', None)
        if fused_conf is not None:
            conf_map = fused_conf[0, 0].cpu().numpy()
        else:
            conf_map = None

        peaks = np.where(detection_map > self.det_threshold)

        for y, x in zip(peaks[0], peaks[1]):
            world_x = (x / self.config.bev_width *
                      (self.config.bev_x_range[1] - self.config.bev_x_range[0]) +
                      self.config.bev_x_range[0])
            world_y = (y / self.config.bev_height *
                      (self.config.bev_y_range[1] - self.config.bev_y_range[0]) +
                      self.config.bev_y_range[0])

            # 体素置信度（论文公式8/9）
            voxel_conf = conf_map[y, x] if conf_map is not None else float(detection_map[y, x])

            detections.append(Detection(
                center=(world_x, world_y),
                confidence=float(detection_map[y, x]),
                fusion_state=FusionState.FUSED,
                voxel_confidence=float(voxel_conf)
            ))

        return detections

    def evaluate(self):
        """运行完整评估。"""
        perframe_stats = []
        tracker_stats = []

        tracker = SequenceMOTATracker()
        prev_scene = None

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc="评估中")
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                radar_points = batch['radar_points']
                if isinstance(radar_points, list):
                    radar_points = [rp.to(self.device) for rp in radar_points]
                else:
                    radar_points = radar_points.to(self.device)

                gt_pos_raw = batch['gt_positions']
                gt_ids_raw = batch['gt_ids']
                if isinstance(gt_pos_raw, list):
                    if len(gt_pos_raw) == 0 or gt_pos_raw[0].size(0) == 0:
                        continue
                    gt_positions = gt_pos_raw[0].cpu().numpy()
                    gt_ids = gt_ids_raw[0].cpu().numpy()
                else:
                    if gt_pos_raw.size(0) == 0:
                        continue
                    gt_positions = gt_pos_raw.cpu().numpy()
                    gt_ids = gt_ids_raw.cpu().numpy()

                if gt_positions.ndim == 1:
                    gt_positions = gt_positions.reshape(1, -1)

                scene_name = batch.get('scene_name', None)

                # 场景切换时重置tracker和时序缓存
                if scene_name is not None:
                    curr_scene = scene_name[0] if isinstance(scene_name, (list, tuple)) else scene_name
                    if curr_scene != prev_scene:
                        tracker = SequenceMOTATracker()
                        self.model.reset_temporal()
                        prev_scene = curr_scene

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

                # === Per-frame MOTA ===
                if len(detections) > 0:
                    pred_positions = np.array([[d.center[0], d.center[1]] for d in detections])
                    pred_ids = np.arange(len(detections))
                else:
                    pred_positions = np.zeros((0, 2), dtype=np.float32)
                    pred_ids = np.zeros((0,), dtype=np.int32)

                mota_pf, motp_pf, stats_pf = compute_mota_motp(
                    gt_positions, gt_ids, pred_positions, pred_ids
                )
                perframe_stats.append(stats_pf)

                # === Tracker-based MOTA（含自适应卡尔曼噪声）===
                tracker.update(detections)
                tracks = tracker.get_confirmed_tracks()
                trk_positions = np.array([t.position for t in tracks]) if tracks else np.zeros((0, 2))
                trk_ids = np.array([t.track_id for t in tracks]) if tracks else np.zeros((0,))

                mota_trk, motp_trk, stats_trk = compute_mota_motp(
                    gt_positions, gt_ids, trk_positions, trk_ids
                )
                tracker_stats.append(stats_trk)

                if batch_idx % 50 == 0:
                    pbar.set_postfix({
                        'MOTA_pf': f'{mota_pf:.3f}',
                        'Dets': len(detections),
                        'Tracks': len(tracks)
                    })

        # === 结果汇总 ===
        pf_mota, pf_motp = accumulate_mota_stats(perframe_stats)
        trk_mota, trk_motp = accumulate_mota_stats(tracker_stats)

        pf_fp = sum(s['FP'] for s in perframe_stats)
        pf_fn = sum(s['FN'] for s in perframe_stats)
        pf_idsw = sum(s['IDSW'] for s in perframe_stats)
        pf_matches = sum(s['matches'] for s in perframe_stats)
        pf_gt = sum(s['num_gt'] for s in perframe_stats)

        trk_fp = sum(s['FP'] for s in tracker_stats)
        trk_fn = sum(s['FN'] for s in tracker_stats)
        trk_idsw = sum(s['IDSW'] for s in tracker_stats)
        trk_matches = sum(s['matches'] for s in tracker_stats)
        trk_gt = sum(s['num_gt'] for s in tracker_stats)

        print(f"\n{'='*60}")
        print(f"评估结果")
        print(f"{'='*60}")
        print(f"\n--- Per-frame MOTA ---")
        print(f"  MOTA: {pf_mota:.4f}")
        print(f"  MOTP: {pf_motp:.4f}")
        print(f"  总GT: {pf_gt}")
        print(f"  匹配: {pf_matches}")
        print(f"  假阳性FP: {pf_fp}")
        print(f"  假阴性FN: {pf_fn}")
        print(f"  ID切换: {pf_idsw}")
        print(f"\n--- Tracker-based MOTA（含自适应卡尔曼） ---")
        print(f"  MOTA: {trk_mota:.4f}")
        print(f"  MOTP: {trk_motp:.4f}")
        print(f"  总GT: {trk_gt}")
        print(f"  匹配: {trk_matches}")
        print(f"  假阳性FP: {trk_fp}")
        print(f"  假阴性FN: {trk_fn}")
        print(f"  ID切换: {trk_idsw}")
        print(f"{'='*60}")

        return pf_mota, pf_motp


def main():
    config = BaseConfig(
        data_root="/mnt/ourDataset_v2/ourDataset_v2_label",
        mapping_csv="/mnt/ourDataset_v2/mapping.csv",
        bev_x_range=(-35.0, 35.0),
        bev_y_range=(0.0, 70.0),
        fusion_method='concat',
        batch_size=1,
        num_workers=4,
        device='cuda'
    )

    # 使用v4最优模型评估
    checkpoint_path = './checkpoints_v4/best_model.pth'

    evaluator = Evaluator(config, checkpoint_path)
    evaluator.evaluate()


if __name__ == '__main__':
    main()
