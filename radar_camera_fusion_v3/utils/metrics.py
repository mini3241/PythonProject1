"""
MOTA/MOTP metrics - imported from existing codebase.
Preserves the original evaluation logic.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def compute_mota_motp(gt_positions: np.ndarray, gt_ids: np.ndarray,
                     pred_positions: np.ndarray, pred_ids: np.ndarray,
                     distance_threshold: float = 2.0) -> Tuple[float, float, Dict[str, int]]:
    """
    Compute MOTA and MOTP metrics.

    Args:
        gt_positions: (N, 2) ground truth positions
        gt_ids: (N,) ground truth IDs
        pred_positions: (M, 2) predicted positions
        pred_ids: (M,) predicted IDs
        distance_threshold: Distance threshold for matching

    Returns:
        mota: MOTA score
        motp: MOTP score
        stats: Dictionary with detailed statistics
    """
    # Ensure arrays are at least 1-dimensional
    gt_positions = np.atleast_2d(gt_positions)
    gt_ids = np.atleast_1d(gt_ids)
    pred_positions = np.atleast_2d(pred_positions)
    pred_ids = np.atleast_1d(pred_ids)

    if len(gt_positions) == 0:
        if len(pred_positions) == 0:
            return 1.0, 0.0, {'FP': 0, 'FN': 0, 'IDSW': 0, 'matches': 0, 'num_gt': 0, 'num_pred': 0}
        else:
            return 0.0, 0.0, {'FP': len(pred_positions), 'FN': 0, 'IDSW': 0, 'matches': 0, 'num_gt': 0, 'num_pred': len(pred_positions)}

    if len(pred_positions) == 0:
        return 0.0, 0.0, {'FP': 0, 'FN': len(gt_positions), 'IDSW': 0, 'matches': 0, 'num_gt': len(gt_positions), 'num_pred': 0}

    # Compute distance matrix
    dist_matrix = np.zeros((len(gt_positions), len(pred_positions)))
    for i, gt_pos in enumerate(gt_positions):
        for j, pred_pos in enumerate(pred_positions):
            dist_matrix[i, j] = np.linalg.norm(gt_pos - pred_pos)

    # Hungarian matching
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Filter matches by distance threshold
    matches = []
    matched_distances = []
    for i, j in zip(row_ind, col_ind):
        if dist_matrix[i, j] <= distance_threshold:
            matches.append((i, j))
            matched_distances.append(dist_matrix[i, j])

    # Compute statistics
    num_matches = len(matches)
    num_fn = len(gt_positions) - num_matches  # False negatives (missed detections)
    num_fp = len(pred_positions) - num_matches  # False positives

    # Compute ID switches
    num_idsw = 0
    gt_to_pred_map = {}
    for gt_idx, pred_idx in matches:
        gt_id = gt_ids[gt_idx]
        pred_id = pred_ids[pred_idx]

        if gt_id in gt_to_pred_map:
            if gt_to_pred_map[gt_id] != pred_id:
                num_idsw += 1
                gt_to_pred_map[gt_id] = pred_id
        else:
            gt_to_pred_map[gt_id] = pred_id

    # Compute MOTA
    num_gt = len(gt_positions)
    mota = 1.0 - (num_fp + num_fn + num_idsw) / num_gt if num_gt > 0 else 0.0

    # Compute MOTP
    motp = np.mean(matched_distances) if len(matched_distances) > 0 else 0.0

    stats = {
        'FP': num_fp,
        'FN': num_fn,
        'IDSW': num_idsw,
        'matches': num_matches,
        'num_gt': num_gt,
        'num_pred': len(pred_positions),
        'matched_distances': matched_distances
    }

    return mota, motp, stats


def accumulate_mota_stats(all_stats: List[Dict[str, int]]) -> Tuple[float, float]:
    """
    Accumulate MOTA statistics across multiple frames.

    Args:
        all_stats: List of statistics dictionaries from each frame

    Returns:
        overall_mota: Overall MOTA score
        overall_motp: Overall MOTP score
    """
    total_fp = sum(s['FP'] for s in all_stats)
    total_fn = sum(s['FN'] for s in all_stats)
    total_idsw = sum(s['IDSW'] for s in all_stats)
    total_gt = sum(s['num_gt'] for s in all_stats)

    if total_gt == 0:
        return 0.0, 0.0

    overall_mota = 1.0 - (total_fp + total_fn + total_idsw) / total_gt

    # MOTP: average distance over all matched pairs across all frames
    all_distances = []
    for s in all_stats:
        all_distances.extend(s.get('matched_distances', []))
    overall_motp = float(np.mean(all_distances)) if len(all_distances) > 0 else 0.0

    return overall_mota, overall_motp
