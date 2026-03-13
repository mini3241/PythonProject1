"""
Evaluation script for radar-camera fusion model.
Preserves original evaluation logic.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion_v2.config.base import BaseConfig
from radar_camera_fusion_v2.models.base_model import RadarCameraFusionModel
from radar_camera_fusion_v2.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion_v2.utils.tracker import SequenceMOTATracker, Detection, FusionState
from radar_camera_fusion_v2.utils.metrics import compute_mota_motp, accumulate_mota_stats


class Evaluator:
    """Evaluation pipeline for radar-camera fusion."""

    def __init__(self, config: BaseConfig, checkpoint_path: str):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = RadarCameraFusionModel(config).to(self.device)

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

        self.model.eval()

        # Create dataset
        valid_list = os.path.join(os.path.dirname(config.mapping_csv), 'valid.txt')
        self.valid_dataset = RadarCameraDataset(config, valid_list, is_train=False)

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )

        # Tracker
        self.tracker = SequenceMOTATracker()

        print(f"Initialized evaluator with {len(self.valid_dataset)} validation samples")

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

    def evaluate(self):
        """Run evaluation on validation set."""
        all_stats = []

        # Reset tracker
        self.tracker = SequenceMOTATracker()

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc="Evaluating")
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)

                # Handle radar_points - can be tensor or list
                radar_points = batch['radar_points']
                if isinstance(radar_points, list):
                    radar_points = [rp.to(self.device) for rp in radar_points]
                else:
                    radar_points = radar_points.to(self.device)

                # Skip batches with empty ground truth
                if len(batch['gt_positions']) == 0 or batch['gt_positions'][0].size(0) == 0:
                    continue

                gt_positions = batch['gt_positions'][0].cpu().numpy()
                gt_ids = batch['gt_ids'][0].cpu().numpy()
                scene_name = batch['scene_name']

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

                # Extract detections
                detections = self._extract_detections(outputs)

                # Update tracker
                self.tracker.update(detections)

                # Get tracked positions
                tracks = self.tracker.get_confirmed_tracks()
                pred_positions = np.array([t.position for t in tracks]) if tracks else np.zeros((0, 2))
                pred_ids = np.array([t.track_id for t in tracks]) if tracks else np.zeros((0,))

                # Compute MOTA/MOTP
                mota, motp, stats = compute_mota_motp(
                    gt_positions, gt_ids,
                    pred_positions, pred_ids
                )
                all_stats.append(stats)

                # Update progress bar
                if batch_idx % 100 == 0:
                    pbar.set_postfix({
                        'MOTA': f'{mota:.3f}',
                        'Dets': len(detections),
                        'Tracks': len(tracks)
                    })

        # Compute overall metrics
        overall_mota, overall_motp = accumulate_mota_stats(all_stats)

        # Print detailed statistics
        total_fp = sum(s['FP'] for s in all_stats)
        total_fn = sum(s['FN'] for s in all_stats)
        total_idsw = sum(s['IDSW'] for s in all_stats)
        total_matches = sum(s['matches'] for s in all_stats)
        total_gt = sum(s['num_gt'] for s in all_stats)

        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Overall MOTA: {overall_mota:.4f}")
        print(f"Overall MOTP: {overall_motp:.4f}")
        print(f"\nDetailed Statistics:")
        print(f"  Total GT: {total_gt}")
        print(f"  Matches: {total_matches}")
        print(f"  False Positives: {total_fp}")
        print(f"  False Negatives: {total_fn}")
        print(f"  ID Switches: {total_idsw}")
        print(f"{'='*60}")

        return overall_mota, overall_motp


def main():
    # Configuration
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

    # Checkpoint path
    checkpoint_path = './checkpoints/latest_model.pth'

    # Create evaluator
    evaluator = Evaluator(config, checkpoint_path)

    # Evaluate
    evaluator.evaluate()


if __name__ == '__main__':
    main()
