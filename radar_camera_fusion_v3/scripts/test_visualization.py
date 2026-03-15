"""
Test script to visualize training data samples with raw data loading.
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, '/mnt/ChillDisk/personal_data/mij/pythonProject1')

from radar_camera_fusion.data.dataset import read_pcd


def load_camera_params(camera_json_path):
    """Load camera intrinsic, extrinsic and distortion from JSON."""
    with open(camera_json_path, 'r') as f:
        data = json.load(f)

    intrinsic = np.array(data['intrinsic'], dtype=np.float32)
    radial_dist = np.array(data.get('radial_distortion', [0, 0, 0]), dtype=np.float32)
    tangential_dist = np.array(data.get('tangential_distortion', [0, 0]), dtype=np.float32)

    # Combine distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([radial_dist[0], radial_dist[1],
                            tangential_dist[0], tangential_dist[1],
                            radial_dist[2]], dtype=np.float32)

    return intrinsic, dist_coeffs


def load_extrinsic(sensor_json_path, sensor_name, camera_name='LeopardCamera0'):
    """Load extrinsic matrix from sensor JSON."""
    with open(sensor_json_path, 'r') as f:
        data = json.load(f)

    extrinsic_key = f'{sensor_name}_to_{camera_name}_extrinsic'
    if extrinsic_key in data:
        return np.array(data[extrinsic_key], dtype=np.float32)
    return None


def project_points_to_image(points, intrinsic, extrinsic, dist_coeffs, img_shape):
    """Project 3D points to image with distortion correction."""
    if len(points) == 0:
        return np.array([]), np.array([])

    # Transform to camera coordinates
    xyz = points[:, :3]
    xyz_homo = np.hstack([xyz, np.ones((len(xyz), 1))])
    xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]

    # Filter points behind camera
    valid_mask = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[valid_mask]

    if len(xyz_cam) == 0:
        return np.array([]), np.array([])

    # Project using cv2.projectPoints (handles distortion)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    image_points, _ = cv2.projectPoints(xyz_cam, rvec, tvec, intrinsic, dist_coeffs)
    image_points = image_points.reshape(-1, 2)

    # Filter within image bounds
    h, w = img_shape[:2]
    u_valid = (image_points[:, 0] >= 0) & (image_points[:, 0] < w)
    v_valid = (image_points[:, 1] >= 0) & (image_points[:, 1] < h)
    valid_mask = u_valid & v_valid

    return image_points[valid_mask], xyz_cam[valid_mask, 2]


def create_depth_map(points, intrinsic, extrinsic, dist_coeffs, img_shape, max_depth=75.0):
    """Create depth map from 3D points."""
    h, w = img_shape[:2]
    depth_map = np.zeros((h, w), dtype=np.float32)

    image_points, depths = project_points_to_image(points, intrinsic, extrinsic, dist_coeffs, img_shape)

    for i in range(len(image_points)):
        u, v = int(image_points[i, 0]), int(image_points[i, 1])
        d = depths[i]
        if 0 <= u < w and 0 <= v < h:
            if depth_map[v, u] == 0 or d < depth_map[v, u]:
                depth_map[v, u] = d

    # Normalize
    depth_map = np.clip(depth_map / max_depth, 0, 1)
    return depth_map


def plot_raw_points_bev(points, gt_positions=None, title="BEV", x_range=(-35, 35), y_range=(0, 70)):
    """Plot raw point cloud xy coordinates in BEV."""
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot points
    if len(points) > 0:
        x = points[:, 0]
        y = points[:, 1]

        # Color by intensity/SNR if available
        if points.shape[1] > 4:
            intensity = points[:, 4]
            scatter = ax.scatter(x, y, c=intensity, cmap='hot', s=5, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Intensity/SNR')
        else:
            ax.scatter(x, y, c='red', s=5, alpha=0.6)

    # Plot GT positions
    if gt_positions is not None and len(gt_positions) > 0:
        gt_x = gt_positions[:, 0]
        gt_y = gt_positions[:, 1]
        ax.scatter(gt_x, gt_y, c='blue', s=100, marker='x', linewidths=3, label='GT')
        ax.legend()

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title}\n{len(points)} points')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def visualize_frame(frame_path, output_path):
    """Visualize a complete frame with all modalities."""

    # Load camera image and params
    camera_path = os.path.join(frame_path, 'LeopardCamera0')
    camera_files = [f for f in os.listdir(camera_path) if f.endswith('.png')]
    if not camera_files:
        print(f"No camera image found in {camera_path}")
        return

    img_file = camera_files[0]
    json_file = img_file.replace('.png', '.json')

    img = cv2.imread(os.path.join(camera_path, img_file))
    intrinsic, dist_coeffs = load_camera_params(os.path.join(camera_path, json_file))

    # Load radar data
    radar_path = os.path.join(frame_path, 'OCULiiRadar')
    radar_pcd_files = [f for f in os.listdir(radar_path) if f.endswith('.pcd')]
    radar_json_files = [f for f in os.listdir(radar_path) if f.endswith('.json')]

    radar_points = np.zeros((0, 5))
    radar_extrinsic = None
    if radar_pcd_files:
        radar_points = read_pcd(os.path.join(radar_path, radar_pcd_files[0]))
    if radar_json_files:
        radar_extrinsic = load_extrinsic(os.path.join(radar_path, radar_json_files[0]), 'OCULiiRadar')

    # Load LiDAR data
    lidar_path = os.path.join(frame_path, 'VelodyneLidar')
    lidar_pcd_files = [f for f in os.listdir(lidar_path) if f.endswith('.pcd')]
    lidar_json_files = [f for f in os.listdir(lidar_path) if f.endswith('.json')]

    lidar_points = np.zeros((0, 5))
    lidar_extrinsic = None
    gt_positions = []
    if lidar_pcd_files:
        lidar_points = read_pcd(os.path.join(lidar_path, lidar_pcd_files[0]))
    if lidar_json_files:
        lidar_json_path = os.path.join(lidar_path, lidar_json_files[0])
        lidar_extrinsic = load_extrinsic(lidar_json_path, 'VelodyneLidar')

        # Load GT annotations (only car class)
        with open(lidar_json_path, 'r') as f:
            lidar_data = json.load(f)
        annotations = lidar_data.get('annotation', [])
        for ann in annotations:
            # Only consider car class
            if ann.get('class') == 'car' and 'x' in ann and 'y' in ann:
                x = ann['x']
                y = ann['y']
                gt_positions.append([x, y])

    gt_positions = np.array(gt_positions) if gt_positions else np.zeros((0, 2))

    # Create visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. RGB Image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'RGB Image\n{img.shape[1]}x{img.shape[0]}')
    ax1.axis('off')

    # 2. LiDAR BEV (raw points)
    ax2 = plt.subplot(2, 4, 2)
    if len(lidar_points) > 0:
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        ax2.scatter(x, y, c='green', s=1, alpha=0.5)
    if len(gt_positions) > 0:
        ax2.scatter(gt_positions[:, 0], gt_positions[:, 1], c='blue', s=100, marker='x', linewidths=3, label='GT')
        ax2.legend()
    ax2.set_xlim(-35, 35)
    ax2.set_ylim(0, 70)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'LiDAR BEV\n{len(lidar_points)} points')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # 3. Radar BEV (raw points with coordinate transformation)
    ax3 = plt.subplot(2, 4, 3)
    if len(radar_points) > 0:
        # OCULii radar coordinate transformation:
        # Original: x=lateral, y=height(±1m), z=forward distance(3-18m)
        # Standard: x=lateral, y=forward distance, z=height
        # Solution: swap y and z columns
        x = radar_points[:, 0]  # lateral (unchanged)
        y = radar_points[:, 2]  # forward distance (was z)
        # z = radar_points[:, 1]  # height (was y) - not used in BEV

        if radar_points.shape[1] > 4:
            scatter = ax3.scatter(x, y, c=radar_points[:, 4], cmap='hot', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax3, label='SNR')
        else:
            ax3.scatter(x, y, c='red', s=10, alpha=0.6)
    if len(gt_positions) > 0:
        ax3.scatter(gt_positions[:, 0], gt_positions[:, 1], c='blue', s=100, marker='x', linewidths=3, label='GT')
        ax3.legend()
    ax3.set_xlim(-35, 35)
    ax3.set_ylim(0, 70)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Radar BEV (coord transformed)\n{len(radar_points)} points')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # 4. Detection GT BEV (raw positions)
    ax4 = plt.subplot(2, 4, 4)
    if len(gt_positions) > 0:
        ax4.scatter(gt_positions[:, 0], gt_positions[:, 1], c='blue', s=200, marker='o', alpha=0.6)
        for i, pos in enumerate(gt_positions):
            ax4.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=8)
    ax4.set_xlim(-35, 35)
    ax4.set_ylim(0, 70)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'Detection GT\n{len(gt_positions)} objects')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # 5. LiDAR Depth
    ax5 = plt.subplot(2, 4, 5)
    if lidar_extrinsic is not None:
        lidar_depth = create_depth_map(lidar_points, intrinsic, lidar_extrinsic, dist_coeffs, img.shape)
        im5 = ax5.imshow(lidar_depth, cmap='jet')
        ax5.set_title(f'LiDAR Depth\nNon-zero: {(lidar_depth>0).sum()}')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
    else:
        ax5.text(0.5, 0.5, 'No extrinsic', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('LiDAR Depth')
    ax5.axis('off')

    # 6. Radar Depth
    ax6 = plt.subplot(2, 4, 6)
    if radar_extrinsic is not None:
        radar_depth = create_depth_map(radar_points, intrinsic, radar_extrinsic, dist_coeffs, img.shape)
        im6 = ax6.imshow(radar_depth, cmap='jet')
        ax6.set_title(f'Radar Depth\nNon-zero: {(radar_depth>0).sum()}')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
    else:
        ax6.text(0.5, 0.5, 'No extrinsic', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Radar Depth')
    ax6.axis('off')

    # 7. LiDAR on Image
    ax7 = plt.subplot(2, 4, 7)
    img_lidar = img.copy()
    if lidar_extrinsic is not None:
        image_points, depths = project_points_to_image(lidar_points, intrinsic, lidar_extrinsic, dist_coeffs, img.shape)
        for i in range(len(image_points)):
            u, v = int(image_points[i, 0]), int(image_points[i, 1])
            color_intensity = int(np.clip(depths[i] / 75.0 * 255, 0, 255))
            cv2.circle(img_lidar, (u, v), 2, (0, 255-color_intensity, color_intensity), -1)
        ax7.set_title(f'LiDAR on Image\n{len(image_points)} projected')
    else:
        ax7.set_title('LiDAR on Image\nNo extrinsic')
    ax7.imshow(cv2.cvtColor(img_lidar, cv2.COLOR_BGR2RGB))
    ax7.axis('off')

    # 8. Radar on Image
    ax8 = plt.subplot(2, 4, 8)
    img_radar = img.copy()
    if radar_extrinsic is not None:
        image_points, depths = project_points_to_image(radar_points, intrinsic, radar_extrinsic, dist_coeffs, img.shape)
        for i in range(len(image_points)):
            u, v = int(image_points[i, 0]), int(image_points[i, 1])
            cv2.circle(img_radar, (u, v), 3, (0, 0, 255), -1)
        ax8.set_title(f'Radar on Image\n{len(image_points)} projected')
    else:
        ax8.set_title('Radar on Image\nNo extrinsic')
    ax8.imshow(cv2.cvtColor(img_radar, cv2.COLOR_BGR2RGB))
    ax8.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")
    print(f"  Camera: {img.shape}, Intrinsic: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
    print(f"  Distortion: k1={dist_coeffs[0]:.3f}, k2={dist_coeffs[1]:.3f}")
    print(f"  LiDAR: {len(lidar_points)} points, Radar: {len(radar_points)} points")
    print(f"  GT: {len(gt_positions)} objects")
    if len(lidar_points) > 0:
        print(f"  LiDAR range: X=[{lidar_points[:,0].min():.1f}, {lidar_points[:,0].max():.1f}], Y=[{lidar_points[:,1].min():.1f}, {lidar_points[:,1].max():.1f}]")
    if len(radar_points) > 0:
        print(f"  Radar original: X=[{radar_points[:,0].min():.1f}, {radar_points[:,0].max():.1f}], Y=[{radar_points[:,1].min():.1f}, {radar_points[:,1].max():.1f}], Z=[{radar_points[:,2].min():.1f}, {radar_points[:,2].max():.1f}]")
        print(f"  Radar BEV (after swap): X=[{radar_points[:,0].min():.1f}, {radar_points[:,0].max():.1f}], Y=[{radar_points[:,2].min():.1f}, {radar_points[:,2].max():.1f}]")


def main():
    data_root = "/mnt/ourDataset_v2/ourDataset_v2_label"
    mapping_csv = "/mnt/ourDataset_v2/mapping.csv"

    # Load mapping
    import pandas as pd
    mapping = pd.read_csv(mapping_csv, header=None, names=['id', 'relpath'])

    # Load train IDs
    train_txt = os.path.join(os.path.dirname(mapping_csv), 'train.txt')
    with open(train_txt, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]

    # Filter valid samples
    samples = mapping[mapping['id'].isin([int(x) for x in train_ids])]

    print(f"Total samples: {len(samples)}")

    output_dir = './visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize specific samples
    for idx in [1000, 2000, 3000]:
        if idx >= len(samples):
            print(f"Sample {idx} out of range")
            continue

        row = samples.iloc[idx]
        frame_path = os.path.join(data_root, row['relpath'])

        print(f"\n=== Sample {idx}: {row['relpath']} ===")

        if not os.path.exists(frame_path):
            print(f"Frame path does not exist: {frame_path}")
            continue

        try:
            output_path = os.path.join(output_dir, f'sample_{idx:04d}.png')
            visualize_frame(frame_path, output_path)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nVisualization complete. Saved to {output_dir}/")


if __name__ == '__main__':
    main()
