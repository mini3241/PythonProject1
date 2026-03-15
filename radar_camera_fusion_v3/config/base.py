"""
Base configuration for radar-camera fusion system.
Clean, modular configuration using dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BaseConfig:
    """Base configuration for the entire system."""

    # Data configuration
    data_root: str = "/mnt/ourDataset_v2/ourDataset_v2_label"
    mapping_csv: str = "/mnt/ourDataset_v2/mapping.csv"

    # BEV configuration
    bev_x_range: Tuple[float, float] = (-35.0, 35.0)  # x_min, x_max
    bev_y_range: Tuple[float, float] = (0.0, 70.0)   # y_min, y_max
    bev_resolution: float = 0.2  # meters per pixel
    bev_height: int = 700  # (bev_y_range[1] - bev_y_range[0]) / bev_resolution
    bev_width: int = 700   # (bev_x_range[1] - bev_x_range[0]) / bev_resolution

    # Radar configuration
    radar_features: List[str] = field(default_factory=lambda: ['x', 'y', 'z', 'doppler', 'snr'])
    voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2)  # x, y, z
    max_points_per_voxel: int = 100
    max_voxels: int = 16000

    # YOLO configuration
    yolo_model_name: str = 'yolov5m'
    yolo_repo_path: str = '/mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/Detection/yolov5'
    yolo_weights_path: str = '/mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/Detection/yolov5/weights/yolov5m.pt'
    yolo_conf_threshold: float = 0.3
    yolo_iou_threshold: float = 0.5
    yolo_classes: List[int] = field(default_factory=lambda: [2])  # car class only

    # Image configuration
    image_size: Tuple[int, int] = (512, 960)
    resnet_pretrained: bool = True

    # Fusion configuration
    fusion_method: str = 'concat'  # 'concat', 'weighted_sum', 'attention'

    # Detection configuration
    enable_detection: bool = True  # Enable detection head for BEV object detection

    # Training configuration
    batch_size: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Device configuration
    device: str = 'cuda'
    cuda_visible_devices: str = '6'

    def __post_init__(self):
        """Post-initialization validation."""
        self.bev_height = int((self.bev_y_range[1] - self.bev_y_range[0]) / self.bev_resolution)
        self.bev_width = int((self.bev_x_range[1] - self.bev_x_range[0]) / self.bev_resolution)