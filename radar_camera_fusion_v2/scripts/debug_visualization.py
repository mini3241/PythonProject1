"""
Debug visualization script to verify data pipeline and model outputs.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion.config.base import BaseConfig
from radar_camera_fusion.data.dataset import RadarCameraDataset, custom_collate_fn
from radar_camera_fusion.models.base_model import RadarCameraFusionModel
from torch.utils.data import DataLoader


def plot_bev_points(points, gt_positions=None, title="BEV", x_range=(-35, 35), y_range=(0, 70)):
    """Plot points in BEV view."""
    fig, ax = plt.subplots(figsize=(6, 8))

    if len(points) > 0:
        x = points[:, 0]
        y = points[:, 1]

        if points.shape[1] > 4:
            intensity = points[:, 4]
            scatter = ax.scatter(x, y, c=intensity, cmap='hot', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Intensity')
        else:
            ax.scatter(x, y, c='red', s=10, alpha=0.6)

    if gt_positions is not None and len(gt_positions) > 0:
        gt_x = gt_positions[:, 0]
        gt_y = gt_positions[:, 1]
        ax.scatter(gt_x, gt_y, c='blue', s=200, marker='x', linewidths=3, label='GT')
        ax.legend()

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title}\n{len(points)} points')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def plot_bev_heatmap(bev_tensor, title="BEV Heatmap", x_range=(-35, 35), y_range=(0, 70)):
    """Plot BEV feature map as heatmap."""
    fig, ax = plt.subplots(figsize=(6, 8))

    # If multi-channel, take max across channels
    if len(bev_tensor.shape) == 3:
        bev_map = bev_tensor.max(dim=0)[0].cpu().numpy()
    else:
        bev_map = bev_tensor.cpu().numpy()

    im = ax.imshow(bev_map, cmap='jet', origin='lower',
                   extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_detection_map(detection_map, gt_positions=None, title="Detection Map",
                       x_range=(-35, 35), y_range=(0, 70), threshold=0.5):
    """Plot detection heatmap with GT overlay."""
    fig, ax = plt.subplots(figsize=(6, 8))

    # Squeeze and convert to numpy
    if isinstance(detection_map, torch.Tensor):
        det_map = detection_map.squeeze().cpu().numpy()
    else:
        det_map = detection_map.squeeze()

    # Apply sigmoid if needed
    det_map = 1 / (1 + np.exp(-det_map))  # sigmoid

    im = ax.imshow(det_map, cmap='hot', origin='lower',
                   extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Detection Score')

    # Overlay GT positions
    if gt_positions is not None and len(gt_positions) > 0:
        gt_x = gt_positions[:, 0]
        gt_y = gt_positions[:, 1]
        ax.scatter(gt_x, gt_y, c='cyan', s=200, marker='x', linewidths=3, label='GT')
        ax.legend()

    # Mark predicted detections
    peaks = np.where(det_map > threshold)
    if len(peaks[0]) > 0:
        # Convert pixel coords to world coords
        h, w = det_map.shape
        pred_y = peaks[0] / h * (y_range[1] - y_range[0]) + y_range[0]
        pred_x = peaks[1] / w * (x_range[1] - x_range[0]) + x_range[0]
        ax.scatter(pred_x, pred_y, c='lime', s=100, marker='o',
                  edgecolors='white', linewidths=2, label='Pred')
        ax.legend()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def draw_yolo_detections(image, detections, color=(0, 0, 255), thickness=2):
    """
    Draw YOLO 2D bounding boxes on image.

    Args:
        image: BGR image (numpy array)
        detections: List of detection dicts with 'bbox', 'confidence', 'class_name'
        color: BGR color for bounding boxes (default: blue for cars)
        thickness: line thickness

    Returns:
        image with bounding boxes drawn
    """
    img_draw = image.copy()

    if detections is None or len(detections) == 0:
        return img_draw

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        conf = det['confidence']
        class_name = det.get('class_name', 'car')

        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f'{class_name} {conf:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, label_size[1] + 10)
        cv2.rectangle(img_draw, (x1, y1_label - label_size[1] - 10),
                     (x1 + label_size[0], y1_label), color, -1)
        cv2.putText(img_draw, label, (x1, y1_label - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_draw


def create_gaussian_heatmap(gt_positions, config):
    """
    Create Gaussian heatmap from ground truth positions (same as training).

    Args:
        gt_positions: (N, 2) ground truth positions [x, y]
        config: BaseConfig object

    Returns:
        heatmap: (H, W) Gaussian heatmap
    """
    heatmap = np.zeros((config.bev_height, config.bev_width), dtype=np.float32)

    if len(gt_positions) == 0:
        return heatmap

    # Gaussian kernel parameters (same as training)
    sigma = 3  # Standard deviation in pixels
    radius = sigma * 3  # 3-sigma rule

    for pos in gt_positions:
        x, y = pos[0], pos[1]

        # Convert to grid coordinates
        norm_x = (x - config.bev_x_range[0]) / (config.bev_x_range[1] - config.bev_x_range[0])
        norm_y = (y - config.bev_y_range[0]) / (config.bev_y_range[1] - config.bev_y_range[0])

        grid_x = int(norm_x * (config.bev_width - 1))
        grid_y = int(norm_y * (config.bev_height - 1))

        if 0 <= grid_x < config.bev_width and 0 <= grid_y < config.bev_height:
            # Apply Gaussian kernel around GT position
            y_min = max(0, grid_y - radius)
            y_max = min(config.bev_height, grid_y + radius + 1)
            x_min = max(0, grid_x - radius)
            x_max = min(config.bev_width, grid_x + radius + 1)

            for gy in range(y_min, y_max):
                for gx in range(x_min, x_max):
                    dist_sq = (gx - grid_x) ** 2 + (gy - grid_y) ** 2
                    value = np.exp(-dist_sq / (2 * sigma ** 2))
                    heatmap[gy, gx] = max(heatmap[gy, gx], value)

    return heatmap


def visualize_lss_projection(image, intrinsic, lidar_to_camera, config, num_samples=500):
    """
    Visualize LSS sampling points projected onto the image.

    Args:
        image: (H, W, 3) BGR image
        intrinsic: (3, 3) camera intrinsic matrix
        lidar_to_camera: (4, 4) lidar to camera extrinsic matrix
        config: BaseConfig object
        num_samples: number of random BEV points to sample and project

    Returns:
        image with projected points drawn
    """
    import random

    img_vis = image.copy()
    H, W = img_vis.shape[:2]

    # Use same sampling as LSS module
    x_min, x_max = config.bev_x_range
    y_min, y_max = config.bev_y_range
    z_min, z_max = -3.0, 5.0
    num_z_layers = 8

    # Create grid exactly like LSS: (bev_height, bev_width, num_z_layers)
    # Sample every N points to reduce visualization density
    sample_stride = 10  # Sample every 50th point
    x_samples = np.linspace(x_min, x_max, config.bev_width)[::sample_stride]
    y_samples = np.linspace(y_min, y_max, config.bev_height)[::sample_stride]
    z_samples = np.linspace(z_min, z_max, num_z_layers)

    points_3d = []
    for z in z_samples:
        for y in y_samples:
            for x in x_samples:
                points_3d.append([x, y, z])

    points_3d = np.array(points_3d, dtype=np.float32)  # (N, 3)

    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # (N, 4)

    # Transform to camera coordinates
    points_cam_homo = (lidar_to_camera @ points_homo.T).T  # (N, 4)
    points_cam = points_cam_homo[:, :3]  # (N, 3)

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_mask]

    if len(points_cam) == 0:
        cv2.putText(img_vis, "No valid points (all behind camera)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img_vis

    # Project to image
    uv_homo = (intrinsic @ points_cam.T).T  # (N, 3)
    uv = uv_homo[:, :2] / (uv_homo[:, 2:3] + 1e-6)  # (N, 2)

    # Filter points within image bounds
    u_valid = (uv[:, 0] >= 0) & (uv[:, 0] < W)
    v_valid = (uv[:, 1] >= 0) & (uv[:, 1] < H)
    valid_mask = u_valid & v_valid

    uv_valid = uv[valid_mask]
    depths_valid = points_cam[valid_mask, 2]

    # Draw projected points with color based on depth
    if len(uv_valid) > 0:
        # Normalize depth for color mapping
        depth_norm = (depths_valid - depths_valid.min()) / (depths_valid.max() - depths_valid.min() + 1e-6)

        for (u, v), d_norm in zip(uv_valid, depth_norm):
            # Color: blue (near) to red (far)
            color = (int(255 * d_norm), 0, int(255 * (1 - d_norm)))
            cv2.circle(img_vis, (int(u), int(v)), 2, color, -1)

        # Add legend
        cv2.putText(img_vis, f"Projected: {len(uv_valid)}/{num_samples} points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_vis, f"Depth: {depths_valid.min():.1f}m - {depths_valid.max():.1f}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img_vis, "No points project into image", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img_vis


def visualize_sample(config, dataset, model, sample_idx, output_dir):
    """Visualize a single sample with all intermediate outputs in one figure."""

    # Get sample from dataset
    sample = dataset[sample_idx]

    # Prepare batch
    batch = custom_collate_fn([sample])

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = batch['images'].to(device)
    radar_points = batch['radar_points']
    if isinstance(radar_points, list):
        radar_points = [rp.to(device) for rp in radar_points]
    else:
        radar_points = radar_points.to(device)

    intrinsic_matrix = batch['intrinsic_matrix'].to(device)
    lidar_to_camera_extrinsic = batch['lidar_to_camera_extrinsic'].to(device)
    gt_positions = batch['gt_positions']
    lidar_depth = batch['lidar_depth']

    print(f"\n=== Sample {sample_idx}: {batch['scene_name']} ===")
    print(f"Image shape: {images.shape}")
    print(f"Radar points: {radar_points[0].shape if isinstance(radar_points, list) else radar_points.shape}")
    print(f"GT positions: {gt_positions[0].shape if isinstance(gt_positions, list) else gt_positions.shape}")
    print(f"Lidar depth shape: {lidar_depth.shape}")

    # Create output directory
    sample_dir = os.path.join(output_dir, f'sample_{sample_idx:04d}')
    os.makedirs(sample_dir, exist_ok=True)

    # Extract data
    img_raw = batch['image_raw']
    if isinstance(img_raw, list):
        img_raw = img_raw[0]

    radar_pts = radar_points[0] if isinstance(radar_points, list) else radar_points
    radar_pts_np = radar_pts.cpu().numpy()
    gt_pos = gt_positions[0] if isinstance(gt_positions, list) else gt_positions
    gt_pos_np = gt_pos.cpu().numpy()
    depth_gt = lidar_depth[0].cpu().numpy()

    # Run model forward pass
    model.eval()
    with torch.no_grad():
        model_input = {
            'images': images,
            'radar_points': radar_points,
            'intrinsic_matrix': intrinsic_matrix,
            'lidar_to_camera_extrinsic': lidar_to_camera_extrinsic
        }
        outputs = model(model_input)

    # Create combined figure with 4x4 grid
    fig = plt.figure(figsize=(26, 24))

    # 1. Input Image
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    ax1.set_title('Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. LSS Projection Visualization
    ax2 = plt.subplot(4, 4, 2)
    # Use original intrinsic (without padding adjustment) for original image
    from radar_camera_fusion.data.dataset import DEFAULT_CAMERA_INTRINSIC
    intrinsic_original = DEFAULT_CAMERA_INTRINSIC
    lidar_to_camera_np = batch['lidar_to_camera_extrinsic'][0].cpu().numpy()
    img_with_projection = visualize_lss_projection(img_raw, intrinsic_original, lidar_to_camera_np, config, num_samples=1000)
    ax2.imshow(cv2.cvtColor(img_with_projection, cv2.COLOR_BGR2RGB))
    ax2.set_title('LSS Projection Points', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. YOLO Detections
    ax2b = plt.subplot(4, 4, 3)
    if 'yolo_detections' in outputs:
        yolo_dets = outputs['yolo_detections']
        if isinstance(yolo_dets, list) and len(yolo_dets) > 0:
            img_with_yolo = draw_yolo_detections(img_raw, yolo_dets[0])
            ax2b.imshow(cv2.cvtColor(img_with_yolo, cv2.COLOR_BGR2RGB))
            ax2b.set_title(f'YOLO Detections ({len(yolo_dets[0])} cars)', fontsize=12, fontweight='bold')
            print(f"YOLO detected {len(yolo_dets[0])} cars")
        else:
            ax2b.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
            ax2b.set_title('YOLO Detections (0 cars)', fontsize=12, fontweight='bold')
    else:
        ax2b.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        ax2b.set_title('YOLO Detections (N/A)', fontsize=12, fontweight='bold')
    ax2b.axis('off')

    # 4. Placeholder for future use
    ax2c = plt.subplot(4, 4, 4)
    ax2c.text(0.5, 0.5, 'Reserved', ha='center', va='center', transform=ax2c.transAxes)
    ax2c.set_title('Reserved', fontsize=12, fontweight='bold')
    ax2c.axis('off')

    # 5. LiDAR Depth GT (moved to row 2)
    ax3 = plt.subplot(4, 4, 5)
    im3 = ax3.imshow(depth_gt, cmap='jet')
    ax3.set_title(f'LiDAR Depth GT\nNon-zero: {(depth_gt > 0).sum()}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 6. Predicted Depth
    ax4 = plt.subplot(4, 4, 6)
    if 'depth_map' in outputs:
        depth_pred = outputs['depth_map'][0, 0].cpu().numpy()
        im4 = ax4.imshow(depth_pred, cmap='jet')
        ax4.set_title('Predicted Depth Map', fontsize=12, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    else:
        ax4.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        ax4.set_title('Predicted Depth Map', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 7. Radar Points BEV
    ax5 = plt.subplot(4, 4, 7)
    if len(radar_pts_np) > 0:
        x, y = radar_pts_np[:, 0], radar_pts_np[:, 1]
        if radar_pts_np.shape[1] > 4:
            sc5 = ax5.scatter(x, y, c=radar_pts_np[:, 4], cmap='hot', s=10, alpha=0.6)
            plt.colorbar(sc5, ax=ax5, fraction=0.046, pad=0.04)
        else:
            ax5.scatter(x, y, c='red', s=10, alpha=0.6)
    if len(gt_pos_np) > 0:
        ax5.scatter(gt_pos_np[:, 0], gt_pos_np[:, 1], c='blue', s=200, marker='x', linewidths=3, label='GT')
        ax5.legend()
    ax5.set_xlim(config.bev_x_range)
    ax5.set_ylim(config.bev_y_range)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title(f'Radar Points BEV\n{len(radar_pts_np)} points', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')

    # 8. Placeholder for future use
    ax5b = plt.subplot(4, 4, 8)
    ax5b.text(0.5, 0.5, 'Reserved', ha='center', va='center', transform=ax5b.transAxes)
    ax5b.set_title('Reserved', fontsize=12, fontweight='bold')
    ax5b.axis('off')

    # 9. Pseudo-LiDAR Points BEV
    ax6 = plt.subplot(4, 4, 9)
    if 'pseudo_points' in outputs:
        pseudo_pts = outputs['pseudo_points']
        if isinstance(pseudo_pts, torch.Tensor) and len(pseudo_pts) > 0:
            pseudo_pts_np = pseudo_pts.cpu().numpy()
            x, y = pseudo_pts_np[:, 0], pseudo_pts_np[:, 1]
            ax6.scatter(x, y, c='red', s=10, alpha=0.6)
            if len(gt_pos_np) > 0:
                ax6.scatter(gt_pos_np[:, 0], gt_pos_np[:, 1], c='blue', s=200, marker='x', linewidths=3, label='GT')
                ax6.legend()
            ax6.set_title(f'Pseudo-LiDAR Points BEV\n{len(pseudo_pts_np)} points', fontsize=12, fontweight='bold')
            print(f"Generated {len(pseudo_pts_np)} pseudo-LiDAR points")
        else:
            ax6.text(0.5, 0.5, 'No pseudo points', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Pseudo-LiDAR Points BEV\n0 points', fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Pseudo-LiDAR Points BEV', fontsize=12, fontweight='bold')
    ax6.set_xlim(config.bev_x_range)
    ax6.set_ylim(config.bev_y_range)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    # 10. Placeholder for future use
    ax6b = plt.subplot(4, 4, 10)
    ax6b.text(0.5, 0.5, 'Reserved', ha='center', va='center', transform=ax6b.transAxes)
    ax6b.set_title('Reserved', fontsize=12, fontweight='bold')
    ax6b.axis('off')

    # 11. Radar BEV Features
    ax7 = plt.subplot(4, 4, 11)
    if 'radar_bev' in outputs:
        bev_map = outputs['radar_bev'][0].max(dim=0)[0].cpu().numpy()
        im7 = ax7.imshow(bev_map, cmap='jet', origin='lower',
                        extent=[config.bev_x_range[0], config.bev_x_range[1],
                               config.bev_y_range[0], config.bev_y_range[1]], aspect='auto')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        ax7.set_title('Radar BEV Features', fontsize=12, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Radar BEV Features', fontsize=12, fontweight='bold')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.grid(True, alpha=0.3)

    # 12. Placeholder for future use
    ax7b = plt.subplot(4, 4, 12)
    ax7b.text(0.5, 0.5, 'Reserved', ha='center', va='center', transform=ax7b.transAxes)
    ax7b.set_title('Reserved', fontsize=12, fontweight='bold')
    ax7b.axis('off')

    # 13. Image BEV Features (LSS)
    ax8 = plt.subplot(4, 4, 13)
    if 'image_bev' in outputs:
        bev_map = outputs['image_bev'][0].max(dim=0)[0].cpu().numpy()
        im8 = ax8.imshow(bev_map, cmap='jet', origin='lower',
                        extent=[config.bev_x_range[0], config.bev_x_range[1],
                               config.bev_y_range[0], config.bev_y_range[1]], aspect='auto')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        ax8.set_title('Image BEV Features (LSS)', fontsize=12, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Image BEV Features (LSS)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.grid(True, alpha=0.3)

    # 14. Pseudo-LiDAR BEV Features
    ax9 = plt.subplot(4, 4, 14)
    if 'pseudo_bev' in outputs:
        bev_map = outputs['pseudo_bev'][0].max(dim=0)[0].cpu().numpy()
        im9 = ax9.imshow(bev_map, cmap='jet', origin='lower',
                        extent=[config.bev_x_range[0], config.bev_x_range[1],
                               config.bev_y_range[0], config.bev_y_range[1]], aspect='auto')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)
        ax9.set_title('Pseudo-LiDAR BEV Features', fontsize=12, fontweight='bold')
    else:
        ax9.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Pseudo-LiDAR BEV Features', fontsize=12, fontweight='bold')
    ax9.set_xlabel('X (m)')
    ax9.set_ylabel('Y (m)')
    ax9.grid(True, alpha=0.3)

    # 10. Fused BEV Features
    ax10 = plt.subplot(4, 4, 15)
    if 'fused_bev' in outputs:
        bev_map = outputs['fused_bev'][0].max(dim=0)[0].cpu().numpy()
        im10 = ax10.imshow(bev_map, cmap='jet', origin='lower',
                          extent=[config.bev_x_range[0], config.bev_x_range[1],
                                 config.bev_y_range[0], config.bev_y_range[1]], aspect='auto')
        plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
        ax10.set_title('Fused BEV Features', fontsize=12, fontweight='bold')
    else:
        ax10.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Fused BEV Features', fontsize=12, fontweight='bold')
    ax10.set_xlabel('X (m)')
    ax10.set_ylabel('Y (m)')
    ax10.grid(True, alpha=0.3)

    # 11. Detection GT BEV (Gaussian Heatmap)
    ax11 = plt.subplot(4, 4, 16)
    if len(gt_pos_np) > 0:
        # Generate Gaussian heatmap (same as training)
        gaussian_heatmap = create_gaussian_heatmap(gt_pos_np, config)
        im11 = ax11.imshow(gaussian_heatmap, cmap='hot', origin='lower',
                          extent=[config.bev_x_range[0], config.bev_x_range[1],
                                 config.bev_y_range[0], config.bev_y_range[1]],
                          aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04, label='Gaussian')

        # Overlay GT positions as markers
        ax11.scatter(gt_pos_np[:, 0], gt_pos_np[:, 1], c='cyan', s=100, marker='x',
                    linewidths=2, label='GT Centers')
        ax11.legend()
        ax11.set_title(f'Detection GT (Gaussian)\n{len(gt_pos_np)} objects', fontsize=12, fontweight='bold')
    else:
        ax11.text(0.5, 0.5, 'No GT', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Detection GT (Gaussian)', fontsize=12, fontweight='bold')
    ax11.set_xlim(config.bev_x_range)
    ax11.set_ylim(config.bev_y_range)
    ax11.set_xlabel('X (m)')
    ax11.set_ylabel('Y (m)')
    ax11.grid(True, alpha=0.3)

    # 12. Predicted Detection Heatmap
    ax12 = plt.subplot(4, 4, 12)
    if 'detection_map' in outputs:
        det_map = outputs['detection_map'][0, 0].cpu().numpy()
        # Apply numerically stable sigmoid to convert logits to probabilities
        # Use piecewise to avoid overflow: sigmoid(x) = 1/(1+exp(-x)) for x >= 0, exp(x)/(1+exp(x)) for x < 0
        det_map_sigmoid = np.zeros_like(det_map)
        positive_mask = det_map >= 0
        negative_mask = det_map < 0
        det_map_sigmoid[positive_mask] = 1.0 / (1.0 + np.exp(-det_map[positive_mask]))
        det_map_sigmoid[negative_mask] = np.exp(det_map[negative_mask]) / (1.0 + np.exp(det_map[negative_mask]))
        im12 = ax12.imshow(det_map_sigmoid, cmap='hot', origin='lower',
                          extent=[config.bev_x_range[0], config.bev_x_range[1],
                                 config.bev_y_range[0], config.bev_y_range[1]],
                          aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04, label='Probability')

        # Overlay GT positions for comparison
        if len(gt_pos_np) > 0:
            ax12.scatter(gt_pos_np[:, 0], gt_pos_np[:, 1], c='cyan', s=100, marker='x',
                        linewidths=2, label='GT Centers')
            ax12.legend()
        ax12.set_title('Predicted Detection Heatmap', fontsize=12, fontweight='bold')
    else:
        ax12.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax12.transAxes)
        ax12.set_title('Predicted Detection Heatmap', fontsize=12, fontweight='bold')
    ax12.set_xlabel('X (m)')
    ax12.set_ylabel('Y (m)')
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(sample_dir, 'combined_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined visualization to {sample_dir}/combined_visualization.png")


def main():
    # Configuration
    config = BaseConfig()

    # Load dataset
    train_list = os.path.join(os.path.dirname(config.mapping_csv), 'train.txt')
    dataset = RadarCameraDataset(config, train_list, is_train=True) #---TRAIN TEST

    print(f"Dataset size: {len(dataset)}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RadarCameraFusionModel(config).to(device)

    # Load checkpoint if exists
    checkpoint_path = './checkpoints/latest_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) #---strict
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found, using randomly initialized model")

    # Output directory
    output_dir = './debug_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize multiple samples
    sample_indices = [0, 10, 20, 1000, 3000]

    for idx in sample_indices:
        if idx >= len(dataset):
            print(f"Sample {idx} out of range")
            continue

        try:
            visualize_sample(config, dataset, model, idx, output_dir)
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDebug visualization complete. Check {output_dir}/")


if __name__ == '__main__':
    main()
