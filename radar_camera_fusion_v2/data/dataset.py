"""
Clean dataset implementation for radar-camera fusion.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional

from ..config.base import BaseConfig

# Default camera intrinsic and extrinsic parameters from LeopardCamera0
DEFAULT_CAMERA_INTRINSIC = np.array([
    [990.1423019264267, 0.0, 479.5298113129775],
    [0.0, 988.780683582274, 249.072061811745],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DEFAULT_CAMERA_EXTRINSIC = np.array([
    [0.9985646158842507, 0.0534044699257085, -0.004082951860174227, 0.1333936485319914],
    [-0.005396032861184197, 0.02446644086851834, -0.9996860887801672, -0.3524396548028612],
    [-0.05328781036315364, 0.9982731869900049, 0.02471949440258006, -0.2159712445341051],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)


def read_pcd(pcd_path: str) -> np.ndarray:
    """Read PCD file and return point cloud data."""
    points = []
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
        data_start = False
        for line in lines:
            if line.startswith('DATA'):
                data_start = True
                continue
            if data_start:
                parts = line.strip().split()
                if len(parts) >= 5:
                    points.append([float(x) for x in parts[:5]])
    return np.array(points, dtype=np.float32)


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_extrinsic(sensor_json_path: str, sensor_name: str, camera_name: str = 'LeopardCamera0') -> Optional[np.ndarray]:
    """Load extrinsic matrix from sensor JSON."""
    data = load_json(sensor_json_path)
    extrinsic_key = f'{sensor_name}_to_{camera_name}_extrinsic'
    if extrinsic_key in data:
        return np.array(data[extrinsic_key], dtype=np.float32)
    return None


def transform_radar_to_lidar(radar_points: np.ndarray, radar_to_camera: np.ndarray, lidar_to_camera: np.ndarray) -> np.ndarray:
    """
    Transform radar points from radar coordinate system to lidar coordinate system.

    Args:
        radar_points: (N, 5) radar points [x, y, z, doppler, snr] in radar coordinate system
        radar_to_camera: (4, 4) extrinsic matrix from radar to camera
        lidar_to_camera: (4, 4) extrinsic matrix from lidar to camera

    Returns:
        transformed_points: (N, 5) radar points in lidar coordinate system
    """
    if len(radar_points) == 0:
        return radar_points

    # Extract xyz coordinates
    xyz = radar_points[:, :3]  # (N, 3)

    # Transform to camera coordinate system
    xyz_homo = np.hstack([xyz, np.ones((len(xyz), 1))])  # (N, 4)
    xyz_camera = (radar_to_camera @ xyz_homo.T).T[:, :3]  # (N, 3)

    # Transform from camera to lidar coordinate system using inverse of lidar_to_camera
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    xyz_camera_homo = np.hstack([xyz_camera, np.ones((len(xyz_camera), 1))])  # (N, 4)
    xyz_lidar = (camera_to_lidar @ xyz_camera_homo.T).T[:, :3]  # (N, 3)

    # Combine with other features (doppler, snr)
    transformed_points = np.hstack([xyz_lidar, radar_points[:, 3:]])  # (N, 5)

    return transformed_points


def pointcloud_to_camera_depth(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray,
                                img_height: int = 640, img_width: int = 640, max_depth: float = 75.0) -> np.ndarray:
    """
    Project 3D point cloud to camera plane to generate depth map.

    Args:
        points: (N, 3+) point cloud [x, y, z, ...]
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) extrinsic matrix (point cloud coord -> camera coord)
        img_height: target image height
        img_width: target image width
        max_depth: maximum depth value for normalization

    Returns:
        depth_map: (img_height, img_width) depth map
    """
    if len(points) == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    # Extract xyz coordinates
    xyz = points[:, :3]  # (N, 3)

    # Transform to camera coordinate system
    xyz_homo = np.hstack([xyz, np.ones((len(xyz), 1))])  # (N, 4)
    xyz_cam = (extrinsic @ xyz_homo.T).T  # (N, 4)
    xyz_cam = xyz_cam[:, :3]  # (N, 3)

    # Filter points behind camera
    valid_mask = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[valid_mask]

    if len(xyz_cam) == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    # Project to image plane
    uv_homo = (intrinsic @ xyz_cam.T).T  # (N, 3)
    uv = uv_homo[:, :2] / (uv_homo[:, 2:3] + 1e-8)  # (N, 2)
    depths = xyz_cam[:, 2]  # (N,)

    # Filter points within image bounds
    u_valid = (uv[:, 0] >= 0) & (uv[:, 0] < img_width)
    v_valid = (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
    valid_mask = u_valid & v_valid

    uv = uv[valid_mask]
    depths = depths[valid_mask]

    if len(uv) == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    # Create depth map
    depth_map = np.zeros((img_height, img_width), dtype=np.float32)
    u_idx = uv[:, 0].astype(np.int32)
    v_idx = uv[:, 1].astype(np.int32)

    # Keep closest depth for each pixel
    for i in range(len(u_idx)):
        u, v, d = u_idx[i], v_idx[i], depths[i]
        if depth_map[v, u] == 0 or d < depth_map[v, u]:
            depth_map[v, u] = d

    # Keep depth in meters (no normalization) for loss computation
    depth_map = np.clip(depth_map, 0, max_depth)

    return depth_map


def simple_pad_to_stride(img: np.ndarray, stride: int = 32,
                         color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image to make dimensions divisible by stride without resizing."""
    h, w = img.shape[:2]

    # Calculate padding needed
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride

    # Pad symmetrically
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img_padded, (top, left)


class RadarCameraDataset(Dataset):
    """Clean dataset for radar-camera fusion."""

    def __init__(self, config: BaseConfig, data_list_file: str, is_train: bool = True):
        """
        Args:
            config: Configuration object
            data_list_file: Path to train.txt or valid.txt
            is_train: Whether this is training dataset
        """
        self.config = config
        self.is_train = is_train
        self.img_size = config.image_size
        self.stride = 32

        # Camera intrinsic and extrinsic parameters
        self.camera_intrinsic = DEFAULT_CAMERA_INTRINSIC
        self.camera_extrinsic = DEFAULT_CAMERA_EXTRINSIC

        # Load mapping from numeric IDs to actual paths
        mapping_file = os.path.join(os.path.dirname(config.data_root), 'mapping.csv')
        self.id_to_path = self._load_mapping(mapping_file)

        # Load data list
        self.data_list = self._load_data_list(data_list_file)

    def _load_mapping(self, mapping_file: str) -> Dict[str, str]:
        """Load mapping from numeric IDs to actual paths."""
        id_to_path = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        id_to_path[parts[0]] = parts[1]
        return id_to_path

    def _load_data_list(self, data_list_file: str) -> List[str]:
        """Load list of data samples from file."""
        with open(data_list_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    def __len__(self) -> int:
        # return 50
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        Returns:
            Dictionary containing:
                - 'image': (C, H, W) preprocessed image tensor
                - 'image_raw': (H, W, C) raw image for visualization
                - 'radar_points': (N, 5) radar points [x, y, z, doppler, snr]
                - 'gt_positions': (M, 2) ground truth positions [x, y]
                - 'gt_ids': (M,) ground truth IDs
                - 'scene_name': str scene identifier
        """
        # Get numeric ID from data list
        numeric_id = self.data_list[idx]

        # Map to actual path (scene_name/frame_name)
        if numeric_id not in self.id_to_path:
            raise ValueError(f"ID {numeric_id} not found in mapping file")

        relative_path = self.id_to_path[numeric_id]
        frame_path = os.path.join(self.config.data_root, relative_path)

        # Load camera image
        camera_path = os.path.join(frame_path, 'LeopardCamera0')
        image_files = [f for f in os.listdir(camera_path) if f.endswith('.png')]
        if not image_files:
            raise ValueError(f"No image found in {camera_path}")

        image_path = os.path.join(camera_path, image_files[0])
        image_raw = cv2.imread(image_path)
        original_h, original_w = image_raw.shape[:2]

        # Preprocess image - simple padding to satisfy 32-divisibility
        image, pad_offset = simple_pad_to_stride(image_raw, stride=self.stride)
        padded_h, padded_w = image.shape[:2]

        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0

        # Load radar points and transform to lidar coordinate system
        radar_path = os.path.join(frame_path, 'OCULiiRadar')
        pcd_files = [f for f in os.listdir(radar_path) if f.endswith('.pcd')]
        json_files = [f for f in os.listdir(radar_path) if f.endswith('.json')]

        if not pcd_files:
            radar_points = np.zeros((0, 5), dtype=np.float32)
        else:
            pcd_path = os.path.join(radar_path, pcd_files[0])
            radar_points = read_pcd(pcd_path)

            # Load radar extrinsic and transform to lidar coordinate system
            if json_files and len(radar_points) > 0:
                radar_json_path = os.path.join(radar_path, json_files[0])
                radar_to_camera = load_extrinsic(radar_json_path, 'OCULiiRadar')

                # We'll load lidar extrinsic later, so store radar_to_camera for now
                radar_extrinsic = radar_to_camera
            else:
                radar_extrinsic = None

        # Load ground truth annotations and LiDAR depth
        lidar_path = os.path.join(frame_path, 'VelodyneLidar')
        # Use padded image size for depth map
        lidar_depth = np.zeros((padded_h, padded_w), dtype=np.float32)
        lidar_to_camera_extrinsic = None

        # Adjust intrinsic matrix for padding offset
        adjusted_intrinsic = self.camera_intrinsic.copy()
        adjusted_intrinsic[0, 2] += pad_offset[1]  # cx += left padding
        adjusted_intrinsic[1, 2] += pad_offset[0]  # cy += top padding

        if os.path.exists(lidar_path):
            json_files = [f for f in os.listdir(lidar_path) if f.endswith('.json')]
            pcd_files = [f for f in os.listdir(lidar_path) if f.endswith('.pcd')]

            # Load annotations and lidar extrinsic
            if json_files:
                json_path = os.path.join(lidar_path, json_files[0])
                lidar_json = load_json(json_path)
                gt_positions, gt_ids = self._parse_annotations(lidar_json)

                # Load lidar extrinsic for radar transformation and LSS
                lidar_to_camera = load_extrinsic(json_path, 'VelodyneLidar')
                lidar_to_camera_extrinsic = lidar_to_camera

                # Transform radar points to lidar coordinate system
                if radar_extrinsic is not None and lidar_to_camera is not None and len(radar_points) > 0:
                    radar_points = transform_radar_to_lidar(radar_points, radar_extrinsic, lidar_to_camera)
            else:
                gt_positions = np.zeros((0, 2), dtype=np.float32)
                gt_ids = np.zeros((0,), dtype=np.int32)

            # Load LiDAR point cloud and generate depth map for padded image
            if pcd_files:
                pcd_path = os.path.join(lidar_path, pcd_files[0])
                lidar_points = read_pcd(pcd_path)
                if len(lidar_points) > 0:
                    lidar_depth = pointcloud_to_camera_depth(
                        lidar_points,
                        adjusted_intrinsic,
                        self.camera_extrinsic,
                        img_height=padded_h,
                        img_width=padded_w,
                        max_depth=75.0
                    )
        else:
            gt_positions = np.zeros((0, 2), dtype=np.float32)
            gt_ids = np.zeros((0,), dtype=np.int32)
            lidar_to_camera_extrinsic = None

        # Use default lidar_to_camera if not loaded
        if lidar_to_camera_extrinsic is None:
            lidar_to_camera_extrinsic = self.camera_extrinsic  # Fallback to camera extrinsic

        return {
            'images': image,
            'image_raw': image_raw,
            'radar_points': torch.from_numpy(radar_points).float(),
            'gt_positions': torch.from_numpy(gt_positions).float(),
            'gt_ids': torch.from_numpy(gt_ids).long(),
            'lidar_depth': torch.from_numpy(lidar_depth).float(),
            'scene_name': relative_path,
            'pad_offset': pad_offset,
            'original_size': (original_h, original_w),
            'padded_size': (padded_h, padded_w),
            'intrinsic_matrix': torch.from_numpy(adjusted_intrinsic).float(),
            'extrinsic_matrix': torch.from_numpy(self.camera_extrinsic).float(),
            'lidar_to_camera_extrinsic': torch.from_numpy(lidar_to_camera_extrinsic).float()
        }

    def _parse_annotations(self, lidar_json: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Parse ground truth annotations from LiDAR JSON (only car class)."""
        annotations = lidar_json.get('annotation', [])

        positions = []
        ids = []

        for ann in annotations:
            # Only consider car class
            if ann.get('class') != 'car':
                continue

            # Extract position - annotations use x, y, z directly
            if 'x' in ann and 'y' in ann:
                x = ann['x']
                y = ann['y']
            else:
                continue

            # Filter by BEV range
            if (self.config.bev_x_range[0] <= x <= self.config.bev_x_range[1] and
                self.config.bev_y_range[0] <= y <= self.config.bev_y_range[1]):
                positions.append([x, y])
                ids.append(ann.get('object_id', -1))

        if positions:
            return np.array(positions, dtype=np.float32), np.array(ids, dtype=np.int32)
        else:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching."""
    # For batch_size=1, just return the first item with batch dimension
    if len(batch) == 1:
        item = batch[0]
        return {
            'images': item['images'].unsqueeze(0),
            'image_raw': item['image_raw'],
            'radar_points': item['radar_points'],
            'gt_positions': item['gt_positions'],
            'gt_ids': item['gt_ids'],
            'lidar_depth': item['lidar_depth'].unsqueeze(0),
            'scene_name': item['scene_name'],
            'pad_offset': item['pad_offset'],
            'original_size': item['original_size'],
            'padded_size': item['padded_size'],
            'intrinsic_matrix': item['intrinsic_matrix'].unsqueeze(0),
            'extrinsic_matrix': item['extrinsic_matrix'].unsqueeze(0),
            'lidar_to_camera_extrinsic': item['lidar_to_camera_extrinsic'].unsqueeze(0)
        }

    # For larger batches, stack appropriately
    images = torch.stack([item['images'] for item in batch])
    lidar_depths = torch.stack([item['lidar_depth'] for item in batch])
    intrinsic_matrices = torch.stack([item['intrinsic_matrix'] for item in batch])
    extrinsic_matrices = torch.stack([item['extrinsic_matrix'] for item in batch])
    lidar_to_camera_extrinsics = torch.stack([item['lidar_to_camera_extrinsic'] for item in batch])

    return {
        'images': images,
        'image_raw': [item['image_raw'] for item in batch],
        'radar_points': [item['radar_points'] for item in batch],
        'gt_positions': [item['gt_positions'] for item in batch],
        'gt_ids': [item['gt_ids'] for item in batch],
        'lidar_depth': lidar_depths,
        'scene_name': [item['scene_name'] for item in batch],
        'pad_offset': [item['pad_offset'] for item in batch],
        'original_size': [item['original_size'] for item in batch],
        'padded_size': [item['padded_size'] for item in batch],
        'intrinsic_matrix': intrinsic_matrices,
        'extrinsic_matrix': extrinsic_matrices,
        'lidar_to_camera_extrinsic': lidar_to_camera_extrinsics
    }
