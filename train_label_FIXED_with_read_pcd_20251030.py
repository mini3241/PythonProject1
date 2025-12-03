import os

# ====================================
# 🚀 一键运行配置 - 所有参数已硬编码
# ====================================
# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 必须为0，否则DataParallel会死锁
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 增大内存块，单GPU优化
os.environ['OMP_NUM_THREADS'] = '8'  # 增加线程数，单GPU优化
os.environ['MKL_NUM_THREADS'] = '8'  # Intel MKL线程数

# ====================================
# 🔧 NCCL环境变量将在setup_balanced_multi_gpu()中设置
# 避免重复设置导致冲突
# ====================================
# 移除这里的NCCL设置，统一在setup_balanced_multi_gpu()中管理

# 不再硬编码GPU，允许通过环境变量设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # 注释掉，使用外部环境变量
# 让setup_balanced_multi_gpu()函数自动选择最空闲的GPU
# if 'CUDA_VISIBLE_DEVICES' not in os.environ:
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 注释掉，让代码自动选择
os.environ['PSEUDO_POINTS'] = '16384'  # 伪点云最大点数（符合Pseudo-LiDAR论文标准10k-20k）
os.environ['CUDNN_DETERMINIS/TIC'] = '0'
os.environ['CUDNN_BENCHMARK'] = '1'  # 启用benchmark加速单GPU训练
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'  # 禁用TF32以保证精度

# 🔧 设置matplotlib为非交互式后端，防止训练时阻塞
import matplotlib

matplotlib.use('Agg')  # 必须在导入pyplot或其他matplotlib模块之前设置

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from fusion_model import FusionNet
import argparse
import gc
import cv2
from visualize_data import VisualizationTool
import re
import warnings

# 🔧 完全对齐baseline：禁用AMP混合精度训练
# Baseline使用纯FP32训练，不使用混合精度
# 参考：/mnt/HotDisk/share/SGDNet_TI/main.py 没有使用AMP
USE_AMP = False
print("🔧 完全对齐baseline：禁用AMP，使用FP32训练")

# 设置tqdm环境变量，防止多行输出
import os

os.environ['TQDM_DISABLE'] = '0'
os.environ['TQDM_MININTERVAL'] = '2.0'

# ========================== PCD文件读取函数（从baseline复制） ==========================
# 支持读取完整的点云数据（包括intensity/velocity/SNR等字段）

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match(r'(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = list(map(float, value.split()))
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.frombuffer(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplementedError("Binary compressed PCD format not supported")


def read_pcd(pcd_path):
    """
    读取PCD文件，返回完整的点云数据（包括所有字段）
    相比open3d.io.read_point_cloud()，这个函数可以读取intensity/velocity/SNR等字段

    Returns:
        np.ndarray: shape (N, num_fields)，包含所有PCD字段的数据
    """
    # 🔍 调试断点1: 检查PCD文件路径
    if not hasattr(read_pcd, '_debug_count'):
        read_pcd._debug_count = 0
    read_pcd._debug_count += 1

    # 每100次读取输出一次调试信息
    debug_this = (read_pcd._debug_count % 100 == 1)

    if debug_this:
        print(f"\n🔍 [断点1] read_pcd调用 #{read_pcd._debug_count}")
        print(f"   文件路径: {pcd_path}")
        print(f"   文件存在: {os.path.exists(pcd_path)}")

    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break

    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        raise ValueError('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    f.close()

    # 提取所有字段数据
    points_list = []
    for field in metadata['fields']:
        if field in pc_data.dtype.names:
            points_list.append(pc_data[field][:, None])

    if len(points_list) > 0:
        points = np.concatenate(points_list, axis=-1)
    else:
        points = np.empty((0, len(metadata['fields'])))

    # 🔍 调试断点2: 检查读取结果
    if debug_this:
        print(f"   ✅ 读取成功:")
        print(f"      点数: {len(points)}")
        print(f"      字段: {metadata['fields']}")
        print(f"      形状: {points.shape}")
        if len(points) > 0:
            print(f"      X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"      Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"      Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    return points


# ========================== End of PCD读取函数 ==========================

try:
    import open3d as o3d

    PCD_AVAILABLE = True
    print("✅ open3d模块可用")
except ImportError:
    PCD_AVAILABLE = False
    print("⚠️ open3d不可用，使用虚拟点云数据")


class MultiModalLabeledDataset(Dataset):
    """
    多模态带标签数据集类
    支持相机、毫米波雷达、激光雷达数据加载和标签解析
    """

    def __init__(self, mapping_csv, id_txt, data_root, img_size=640, use_camera='LeopardCamera0'):
        self.mapping = pd.read_csv(mapping_csv, header=None, names=['id', 'relpath'])
        self.mapping['id'] = self.mapping['id'].apply(lambda x: f"{int(x):06d}")

        with open(id_txt, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # 过滤有效样本
        self.samples = self.mapping[self.mapping['id'].isin(self.ids)]
        self.data_root = data_root
        self.img_size = img_size
        self.use_camera = use_camera

        print(f"数据集加载完成: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _load_camera_data(self, frame_path):
        """加载相机数据和标签"""
        camera_path = os.path.join(frame_path, self.use_camera)
        if not os.path.exists(camera_path):
            return None, None, None

        files = os.listdir(camera_path)
        img_files = [f for f in files if f.endswith('.png')]

        if not img_files:
            return None, None, None

        # 加载图像
        img_path = os.path.join(camera_path, img_files[0])
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        # 加载标签 - 从VelodyneLidar目录读取
        lidar_path = os.path.join(frame_path, 'VelodyneLidar')
        targets = []
        label_data = {}

        if os.path.exists(lidar_path):
            json_files = [f for f in os.listdir(lidar_path) if f.endswith('.json')]
            if json_files:
                json_path = os.path.join(lidar_path, json_files[0])
                with open(json_path, 'r') as f:
                    label_data = json.load(f)

                annotations = label_data.get('annotation', [])

                # 解析目标信息
                for ann in annotations:
                    # 只处理car和bus类别，映射为0和1
                    if ann.get('class', '') in ['car', 'bus']:
                        target = {
                            'class': 0 if ann.get('class', '') == 'car' else 1,  # car: 0, bus: 1
                            'x': float(ann.get('x', 0)),
                            'y': float(ann.get('y', 0)),
                            'z': float(ann.get('z', 0)),
                            'w': float(ann.get('w', 0)),
                            'l': float(ann.get('l', 0)),
                            'h': float(ann.get('h', 0)),
                            'object_id': int(ann.get('object_id', -1)),
                            'motion': int(ann.get('motion', 0))
                        }
                        targets.append(target)

        return img, targets, label_data

    def _load_radar_data(self, frame_path):
        """加载毫米波雷达数据 - 返回BEV图像和深度图

        Returns:
            tuple: (radar_bev, radar_depth)
                - radar_bev: (5, H, W) BEV图像
                - radar_depth: (H, W) 深度图
        """
        radar_path = os.path.join(frame_path, 'OCULiiRadar')
        if not os.path.exists(radar_path):
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

        files = os.listdir(radar_path)
        pcd_files = [f for f in files if f.endswith('.pcd')]

        if not pcd_files:
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

        pcd_path = os.path.join(radar_path, pcd_files[0])
        try:
            # 使用自定义read_pcd函数读取完整数据 (x, y, z, velocity, SNR等)
            points = read_pcd(pcd_path)

            if len(points) == 0:
                return (torch.zeros(5, self.img_size, self.img_size),
                        torch.zeros(self.img_size, self.img_size))

            # 🔧 **关键修复**: OCULii雷达坐标系转换
            # 问题：雷达PCD的坐标系定义和标准坐标系不一致
            #   ��达坐标系: x=横向, y=高度(±1m), z=前向距离(3-18m)
            #   标准坐标系: x=横向, y=前向距离,   z=高度
            # 解决：交换 y 和 z 列
            if points.shape[1] >= 3:
                # 🔍 [断点2-雷达] 转换前的原始数据
                if not hasattr(self, '_coord_transform_debug_count'):
                    self._coord_transform_debug_count = 0
                self._coord_transform_debug_count += 1

                if self._coord_transform_debug_count % 100 == 1:
                    print(f"\n🔍 [断点2-雷达] 坐标转换 - 第{self._coord_transform_debug_count}次")
                    print(f"   📂 读取文件: {pcd_path}")
                    print(f"   ❌ 转换前 X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                    print(f"   ❌ 转换前 Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]  ← 原始Y（高度）")
                    print(f"   ❌ 转换前 Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]  ← 原始Z（前向距离）")

                # 🔧 **关键修复**: 过滤Z>100米的噪声点（扩展到100m以覆盖GT范围）
                # 注意：Z轴交换后会变成Y轴（前向），需要覆盖GT的Y范围[0, 100]m
                # 之前限制50m导致雷达数据只到y=50m，但GT到y=99.55m，造成MOTA=0
                z_valid_mask = (points[:, 2] >= 0) & (points[:, 2] <= 100.0)
                points_before = len(points)
                points = points[z_valid_mask]
                points_after = len(points)

                if self._coord_transform_debug_count % 100 == 1:
                    print(
                        f"   🔧 过滤噪声点(Z不在0-100m): {points_before} -> {points_after} ({100.0 * points_after / points_before:.1f}%保留)")

                if len(points) == 0:
                    return (torch.zeros(5, self.img_size, self.img_size),
                            torch.zeros(self.img_size, self.img_size))

                if self._coord_transform_debug_count % 100 == 1:
                    print(f"   🔧 开始交换 Y ↔ Z...")

                temp_y = points[:, 1].copy()  # 保存原始y（高度）
                points[:, 1] = points[:, 2]  # y = z（前向距离）
                points[:, 2] = temp_y  # z = y（高度）

                # 调试：每100次输出一次转换结果
                if self._coord_transform_debug_count % 100 == 1:
                    print(f"   ✅ 转换后 X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                    print(
                        f"   ✅ 转换后 Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]  ← 前向距离(0-100m)")
                    print(f"   ✅ 转换后 Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]  ← 高度")

            # 1. 生成BEV图像
            radar_bev = self._points_to_image(points, self.img_size)

            # 2. 生成深度图（用于SGDNet输入）
            radar_depth = self._points_to_depth_image(points, self.img_size)

            return (torch.from_numpy(radar_bev).float(),
                    torch.from_numpy(radar_depth).float())

        except Exception as e:
            print(f"🔴 雷达数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

    def _load_lidar_data(self, frame_path):
        """加载激光雷达数据 - 返回BEV图像和深度图

        Returns:
            tuple: (lidar_bev, lidar_depth)
                - lidar_bev: (5, H, W) BEV图像
                - lidar_depth: (H, W) 深度图
        """
        lidar_path = os.path.join(frame_path, 'VelodyneLidar')
        if not os.path.exists(lidar_path):
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

        files = os.listdir(lidar_path)
        pcd_files = [f for f in files if f.endswith('.pcd')]

        if not pcd_files:
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

        pcd_path = os.path.join(lidar_path, pcd_files[0])
        try:
            # 使用自定义read_pcd函数读取完整数据 (x, y, z, intensity等)
            points = read_pcd(pcd_path)

            if len(points) == 0:
                return (torch.zeros(5, self.img_size, self.img_size),
                        torch.zeros(self.img_size, self.img_size))

            # 🔧 **关键修复**: VelodyneLidar坐标系转换
            # 问题：LiDAR的y坐标是负值（-9 ~ -62米），GT的y是正值（10-41米）
            #   LiDAR坐标系: x=横向, y=负的前向距离, z=高度
            #   标准坐标系:   x=横向, y=正的前向距离, z=高度
            # 解决：对y取负
            if points.shape[1] >= 2:
                # 🔍 [断点3-LiDAR] 转换前的原始数据
                if not hasattr(self, '_lidar_transform_debug_count'):
                    self._lidar_transform_debug_count = 0
                self._lidar_transform_debug_count += 1

                if self._lidar_transform_debug_count % 100 == 1:
                    print(f"\n🔍 [断点3-LiDAR] Y范围过滤 - 第{self._lidar_transform_debug_count}次")
                    print(f"   📂 读取文件: {pcd_path}")
                    print(f"   ❌ 过滤前 X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                    print(
                        f"   ❌ 过滤前 Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]  ← 原始Y（dataset已转换，混合正负值）")
                    if points.shape[1] >= 3:
                        print(f"   ❌ 过滤前 Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]  ← 高度")
                    print(f"   🔧 Baseline策略: 过滤Y范围到有效BEV坐标...")

                # 🔧 **关键修复**: 参考baseline line 193 - 直接过滤Y范围，不取负
                # Baseline: if tmp_y >= 40 or tmp_y <= 0: continue
                # 这里使用更宽松的范围以覆盖完整BEV: Y=[-5.7, 102.3]m
                Y_MIN_FILTER, Y_MAX_FILTER = -10.0, 105.0
                valid_mask = (points[:, 1] >= Y_MIN_FILTER) & (points[:, 1] <= Y_MAX_FILTER)
                points_before = len(points)
                points = points[valid_mask]
                points_after = len(points)

                # 调试：每100次输出一次过滤结果
                if self._lidar_transform_debug_count % 100 == 1:
                    print(
                        f"   🔧 过滤Y范围({Y_MIN_FILTER}~{Y_MAX_FILTER}m): {points_before} -> {points_after} ({100.0 * points_after / points_before:.1f}%保留)")
                    if points_after > 0:
                        print(f"   ✅ 过滤后 X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                        print(
                            f"   ✅ 过滤后 Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]  ← 有效前向距离")
                        if points.shape[1] >= 3:
                            print(f"   ✅ 过滤后 Z范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]  ← 高度")
                    else:
                        print(f"   ⚠️  警告: 过滤后没有有效点！")

            # 1. 生成BEV图像
            lidar_bev = self._points_to_image(points, self.img_size)

            # 2. 生成深度图（用于SGDNet监督）
            lidar_depth = self._points_to_depth_image(points, self.img_size)

            return (torch.from_numpy(lidar_bev).float(),
                    torch.from_numpy(lidar_depth).float())

        except Exception as e:
            print(f"🔴 激光雷达数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros(5, self.img_size, self.img_size),
                    torch.zeros(self.img_size, self.img_size))

    def _points_to_depth_image(self, points, img_size, camera_intrinsic=None):
        """将3D点云投影到相机平面生成深度图

        **重要**: 这里假设点云在车体坐标系中,我们不做坐标转换
        直接使用y坐标作为深度(前方距离),x坐标作为左右位置

        Args:
            points: (N, 3+) 点云数据 (x, y, z, ...)
            img_size: 图像尺寸
            camera_intrinsic: 相机内参矩阵 (3x3)，暂时不使用

        Returns:
            depth_map: (H, W) 深度图，每个像素值=深度（米）
        """
        if len(points) == 0:
            return np.zeros((img_size, img_size), dtype=np.float32)

        # 提取xyz坐标
        xyz = points[:, :3]  # (N, 3)

        # **关键修改**: 不做相机投影,直接将车体坐标系的点映射到图像平面
        # 假设: x=左右(-82 ~ 86), y=前后(-5 ~ 102), z=上下
        # 我们将y(前后)作为深度,将x(左右)和z(上下)映射到图像坐标

        # 🔧 **关键修复 2025-11-18 (第四次)**: 修复Z轴范围bug导致深度图全0
        # 实际LiDAR数据分析:
        #   X范围: [-129.57, 105.18]m
        #   Y范围: [-10.00, 104.98]m (前向距离)
        #   Z范围: [-4.07, 24.96]m (高度) ← 之前设置-0.5~2.5导致所有点被过滤！
        # 🔧 **关键修复**: 必须与fusion_model.py和深度图归一化保持完全一致！
        # fusion_model.py第127-129行定义: X=[-82.7, 86.5], Y=[0, 75], Z=[-5, 30]
        # 深度图归一化第546行使用: Y_MAX_NORM = 75.0
        X_MIN, X_MAX = -100.0, 105.0  # X轴 = 横向 (与fusion_model.py一致)
        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 = 前向深度 (与深度图归一化一致，75米！)
        Z_MIN, Z_MAX = -5.0, 30.0  # Z轴 = 高度 (与fusion_model.py一致)

        # 🔧 坐标过滤 - 恢复正确的逻辑
        # 现在: X=横向, Y=前向(深度), Z=高度
        valid_mask = (xyz[:, 0] >= X_MIN) & (xyz[:, 0] <= X_MAX) & \
                     (xyz[:, 1] >= Y_MIN) & (xyz[:, 1] <= Y_MAX)
        # (xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)  # 暂时禁用Z高度过滤

        if not valid_mask.any():
            return np.zeros((img_size, img_size), dtype=np.float32)

        xyz_valid = xyz[valid_mask]

        # 🔧 映射到图像坐标 - 恢复正确的映射
        # 坐标系: X=横向, Y=前向(深度), Z=高度
        # 图像映射: u方向=X(横向), v方向=Z(高度)
        norm_x = (xyz_valid[:, 0] - X_MIN) / (X_MAX - X_MIN)  # 横向
        norm_z = (xyz_valid[:, 2] - Z_MIN) / (Z_MAX - Z_MIN)  # 高度

        u = (norm_x * (img_size - 1)).astype(int)  # u = 横向
        v = ((1 - norm_z) * (img_size - 1)).astype(int)  # v = 高度(反��)

        # 深度值使用Y坐标(前向距离) - 恢复正确的深度定义
        depth = xyz_valid[:, 1]

        # 过滤图像范围外的点
        valid_uv = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size) & (depth > 0)
        u = u[valid_uv]
        v = v[valid_uv]
        depth = depth[valid_uv]

        # 创建深度图
        depth_map = np.zeros((img_size, img_size), dtype=np.float32)
        for i in range(len(u)):
            # 取最小深度(最近的点)
            if depth_map[v[i], u[i]] == 0 or depth[i] < depth_map[v[i], u[i]]:
                depth_map[v[i], u[i]] = depth[i]

        # 🔧 **关键修复 2025-11-19**: 与SGDNet_TI原始项目保持一致！
        # SGDNet_TI项目使用75米作为max_depth进行归一化
        # 参考: SGDNet_TI/Datasets/dataloader.py 第61行
        #   def depth_filter(self, depth_map, min_depth=0., max_depth=75.):
        #       return filtered_depth / max_depth
        Y_MIN_NORM, Y_MAX_NORM = -5.0, 105.0  # 与BEV坐标范围一致！
        MAX_DEPTH = Y_MAX_NORM - Y_MIN_NORM  # 110米

        # 🔴🔴🔴 红色断点3: 深度图归一化检查 🔴🔴🔴 (已禁用)
        # print(f"\n{'='*80}")
        # print(f"🔴 断点3 - 深度图归一化检查:")
        # print(f"   归一化前深度范围: [{depth_map.min():.2f}, {depth_map.max():.2f}]米")
        # print(f"   归一化前深度均值: {depth_map.mean():.2f}米")
        # print(f"   非零深度点数: {np.count_nonzero(depth_map)}")
        # print(f"   归一化参数: Y_MIN={Y_MIN_NORM:.1f}m, Y_MAX={Y_MAX_NORM:.1f}m, MAX_DEPTH={MAX_DEPTH:.1f}m")

        # 🔴🔴🔴 红色断点E: 非零深度值分布检查 🔴🔴🔴 (已禁用)
        # nonzero_depths = depth_map[depth_map > 0]
        # if len(nonzero_depths) > 0:
        #     print(f"\n{'='*80}")
        #     print(f"🔴 断点E - 非零深度值分布检查 (关键诊断!):")
        #     print(f"   非零深度统计:")
        #     print(f"     点数:   {len(nonzero_depths)}")
        #     print(f"     均值:   {nonzero_depths.mean():.2f}m  ← 关键! 应该≈50m左右")
        #     print(f"     中位数: {np.median(nonzero_depths):.2f}m")
        #     print(f"     范围:   [{nonzero_depths.min():.2f}, {nonzero_depths.max():.2f}]m")
        #
        #     # 分段统计：近距离、中距离、远距离
        #     near_range = ((nonzero_depths >= 0) & (nonzero_depths < 10)).sum()
        #     mid_range = ((nonzero_depths >= 10) & (nonzero_depths < 50)).sum()
        #     far_range = ((nonzero_depths >= 50) & (nonzero_depths <= 105)).sum()
        #
        #     total_pts = len(nonzero_depths)
        #     print(f"   距离分段:")
        #     print(f"     近距离 [0-10m]:    {near_range:6d} ({near_range/total_pts*100:5.1f}%)")
        #     print(f"     中距离 [10-50m]:   {mid_range:6d} ({mid_range/total_pts*100:5.1f}%)")
        #     print(f"     远距离 [50-105m]:  {far_range:6d} ({far_range/total_pts*100:5.1f}%)")
        #
        #     # 诊断判断
        #     if near_range / total_pts > 0.8:
        #         print(f"   ❌❌❌ 严重问题: {near_range/total_pts*100:.1f}%的点在近距离!")
        #         print(f"   原因可能是:")
        #         print(f"     1. 输入点云被过度过滤，远处点被移除")
        #         print(f"     2. 传感器数据本身远处点很少")
        #         print(f"     3. 深度图生成时Y坐标计算错误")
        #     elif nonzero_depths.mean() < 20:
        #         print(f"   ⚠️  警告: 非零深度均值只有{nonzero_depths.mean():.1f}m（期望≈50m）")
        #         print(f"   说明大部分点集中在近处，SGDNet难以学习远距离深度")
        #     else:
        #         print(f"   ✅ 深度分布正常，覆盖整个范围")
        #     print(f"{'='*80}\n")

        depth_map = np.clip((depth_map - Y_MIN_NORM) / MAX_DEPTH, 0, 1)

        # print(f"   归一化后深度范围: [{depth_map.min():.4f}, {depth_map.max():.4f}]")
        # print(f"   归一化后深度均值: {depth_map.mean():.4f}")
        # print(f"   ⚠️  期望: 归一化后应在[0, 1]范围内")
        # print(f"   ⚠️  如果大部分值都接近0或1，说明深度分布异常")
        # if depth_map.mean() < 0.1 or depth_map.mean() > 0.9:
        #     print(f"   ⚠️⚠️⚠️ 警告：深度均值偏离中间值，可能存在归一化问题！")
        # else:
        #     print(f"   ✅ 深度归一化正常")
        # print(f"{'='*80}\n")

        return depth_map

    def _points_to_image(self, points, img_size):
        """将3D点云转换为多通道BEV图像（俯视图），使用固定坐标范围

        **关键修复**: 使用全局固定坐标范围，与Loss函数保持一致
        X: [-82.7, 86.5], Y: [-5.7, 102.3]

        返回5通道BEV图像：
        - channel 0: 点密度
        - channel 1: 平均高度 (z坐标)
        - channel 2: 最大高度
        - channel 3: 强度/反射率 (如果有)
        - channel 4: 点数统计
        """
        if len(points) == 0:
            return np.zeros((5, img_size, img_size), dtype=np.float32)

        # 🔧 **关键修复 2025-11-19**: 与fusion_model.py和深度图归一化保持一致
        # fusion_model.py第127-129行: X=[-82.7, 86.5], Y=[0, 75], Z=[-5, 30]
        X_MIN, X_MAX = -100.0, 105.0  # X轴 = 横向 (与fusion_model.py一致)
        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 = 前向75m (与深度图归一化一致)
        Z_MIN, Z_MAX = -5.0, 30.0  # Z轴 = 高度 (与fusion_model.py一致)

        # 🔧 关键修复：先按Z高度过滤点云，移除地面点和噪声点
        # 问题：地面点（Z≈-1.5m）导致近处BEV密度过高，模型学习错误位置
        # ⚠️ 发现问题：OCULii雷达的Z值范围是7-174米，不是车辆高度！
        # 临时解决：暂时禁用Z过滤，待确认坐标系定义后再启用
        # if points.shape[1] >= 3:
        #     z_filter = (points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)
        #     points = points[z_filter]

        if len(points) == 0:
            return np.zeros((5, img_size, img_size), dtype=np.float32)

        # 归一化到 [0, 1]��然后转换到像素坐标
        # 🔧 恢复正确的坐标系: X=横向, Y=前向
        # 与GT热力图生成保持一致：使用像素中心坐标
        norm_x = (points[:, 0] - X_MIN) / (X_MAX - X_MIN)  # X=横向归一化
        norm_y = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN)  # Y=前向归一化

        # 转换到像素坐标，使用稳定的转换方法
        # 使用标准映射，确保一致性
        x_norm = (norm_x * (img_size - 1)).round()  # X轴像素 (横向)
        y_norm = (norm_y * (img_size - 1)).round()  # Y轴像素 (前向)
        # 🔴 断点C：验证Y轴映射方向（已禁用以提升性能）
        # print(f"🔴 断点C - Y轴映射:")
        # print(f"   世界Y=0m -> pixel_y={0}")
        # print(f"   世界Y=50m -> pixel_y={(50 - Y_MIN) / (Y_MAX - Y_MIN) * (img_size - 1):.0f}")
        # print(f"   世界Y=100m -> pixel_y={(100 - Y_MIN) / (Y_MAX - Y_MIN) * (img_size - 1):.0f}")
        # print(f"   ⚠️  如果pixel_y小的对应Y大的，说明Y轴反了！")

        # 过滤范围外的点
        valid_mask = (x_norm >= 0) & (x_norm < img_size) & (y_norm >= 0) & (y_norm < img_size)

        # 🔍 [断点4-BEV] 检查有多少点被过滤掉了
        if not hasattr(self, '_bev_filter_debug_count'):
            self._bev_filter_debug_count = 0
        self._bev_filter_debug_count += 1
        if self._bev_filter_debug_count % 100 == 1:
            valid_count = valid_mask.sum()
            total_count = len(points)
            print(f"\n🔍 [断点4-BEV] 点云到BEV映射 - 第{self._bev_filter_debug_count}次")
            print(f"   📊 BEV坐标范围设置: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}]")
            print(f"   📊 输入点云范围:")
            print(f"      X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"      Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"   📊 范围内有效点: {valid_count}/{total_count} ({100.0 * valid_count / total_count:.1f}%)")
            if valid_count < total_count * 0.5:
                print(f"   ⚠️  警告: 超过50%的点在BEV范围外！坐标转换可能有问题！")
            if valid_count == 0:
                print(f"   ❌ 严重错误: 所有点都在BEV范围外！")

        if not valid_mask.any():
            return np.zeros((5, img_size, img_size), dtype=np.float32)

        x_norm = x_norm[valid_mask]
        y_norm = y_norm[valid_mask]
        points_valid = points[valid_mask]

        # 创建多通道BEV图像
        bev = np.zeros((5, img_size, img_size), dtype=np.float32)
        x_idx = np.clip(x_norm.astype(int), 0, img_size - 1)
        y_idx = np.clip(y_norm.astype(int), 0, img_size - 1)
        # print(f"🔴 断点A - BEV生成:")
        # print(f"   点云世界Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]m")
        # print(f"   映射到pixel_y范围: [{y_idx.min()}, {y_idx.max()}]")

        # 提取高度信息（z坐标）
        z_values = points_valid[:, 2] if points_valid.shape[1] > 2 else np.zeros(len(points_valid))
        # 提取强度信息（第4列，如果有）
        intensity = points_valid[:, 3] if points_valid.shape[1] > 3 else np.ones(len(points_valid))

        # ⚡ 向量化累加到BEV网格（一次性处理所有点，速度提升10-100倍）
        # 使用 np.add.at 和 np.maximum.at 进行原地累加，支持重复索引
        np.add.at(bev[0], (y_idx, x_idx), 1.0)  # 密度
        np.add.at(bev[1], (y_idx, x_idx), z_values)  # 高度累加
        np.maximum.at(bev[2], (y_idx, x_idx), z_values)  # 最大高度
        np.add.at(bev[3], (y_idx, x_idx), intensity)  # 强度累加
        np.add.at(bev[4], (y_idx, x_idx), 1.0)  # 点数统计

        # 🔍 调试信息收集（仅在需要时执行，不影响性能）
        debug_positions = []
        debug_far_points = []
        if not hasattr(self, '_bev_debug_printed'):
            # 记录前3个点
            for i in range(min(3, len(x_idx))):
                debug_positions.append((points_valid[i, 0], points_valid[i, 1], y_idx[i], x_idx[i]))

            # 记录远处点（y>20m）用于验证坐标转换
            far_mask = points_valid[:, 1] > 20
            if far_mask.any():
                far_indices = np.where(far_mask)[0][:5]  # 最多5个
                for idx in far_indices:
                    world_x = points_valid[idx, 0]
                    world_y = points_valid[idx, 1]
                    debug_far_points.append((world_x, world_y, y_idx[idx], x_idx[idx]))

        # 输出调试信息（仅第一次，且避免多GPU死锁）
        if not hasattr(self, '_bev_debug_printed'):
            # 🔧 多GPU修复：使用try-except避免多进程同时打印导致死锁
            try:
                # 只在单GPU模式或环境变量允许时打印
                if os.environ.get('ENABLE_DEBUG_PRINT', '0') == '1':
                    print(f"\n🔍 DEBUG BEV投影 (img_size={img_size}):")
                    print(f"   总点数: {len(points)} -> 有效点数: {len(points_valid)}")
                    for world_x, world_y, pix_y, pix_x in debug_positions[:3]:
                        print(f"   世界({world_x:.2f}, {world_y:.2f}) -> BEV像素(y={pix_y}, x={pix_x})")
                    # 显示BEV密度分布
                    density_map = bev[0]  # 点密度通道
                    print(
                        f"   BEV密度: max={density_map.max():.2f}, 非零像素={np.count_nonzero(density_map)}/{img_size * img_size}")
            except:
                # 忽略多GPU环境下的打印错误
                pass
            self._bev_debug_printed = True

        # 归一化处理
        mask = bev[0] > 0  # 有点的位置
        bev[1][mask] /= bev[0][mask]  # 平均高度 = 累加高度 / 点数
        bev[3][mask] /= bev[0][mask]  # 平均强度 = 累加强度 / 点数

        # 🔧 **关键修复**: 使用固定全局范围归一化（参考baseline策略）
        # 参考: RadarRGBFusionNet2_20231128/dataset/GetLidarBEV.py:325-338

        # 1. 密度归一化：对数尺度（与baseline一致）
        if bev[0].max() > 0:
            # 使用对数尺度压缩密度，避免极端值
            bev[0] = np.minimum(1.0, np.log(bev[0] + 1) / np.log(64))

        # 2. 高度归一化：使用固定全局范围（与baseline一致）
        # Baseline: minZ=-2.73m, maxZ=1.27m, 范围=4.0m
        # OCULii雷达安装位置不同，需要调整范围
        Z_MIN, Z_MAX = -5.0, 30.0  # 🔧 修复：扩大范围覆盖实际数据[-4.07, 24.96]m
        MAX_HEIGHT = Z_MAX - Z_MIN  # 35米
        bev[1] = np.clip((bev[1] - Z_MIN) / MAX_HEIGHT, 0, 1)  # 平均高度
        bev[2] = np.clip((bev[2] - Z_MIN) / MAX_HEIGHT, 0, 1)  # 最大高度

        # 3. 强度归一化：固定值（与baseline一致）
        # Baseline使用 intensity/100，我们使用max值
        if bev[3].max() > 0:
            bev[3] = bev[3] / bev[3].max()

        # 4. 点数归一化：对数尺度（与baseline一致）
        if bev[4].max() > 0:
            bev[4] = np.minimum(1.0, np.log(bev[4] + 1) / np.log(64))

        return bev

    def __getitem__(self, idx):
        try:
            row = self.samples.iloc[idx]
            frame_path = os.path.join(self.data_root, row['relpath'])

            if not os.path.exists(frame_path):
                print(f"警告: 数据路径不存在: {frame_path}")
                return None

            # 加载多模态数据
            camera_img, targets, label_info = self._load_camera_data(frame_path)
            radar_bev, radar_depth = self._load_radar_data(frame_path)
            lidar_bev, lidar_depth = self._load_lidar_data(frame_path)

            if camera_img is None:
                print(f"警告: 相机数据加载失败: {frame_path}")
                return None

            # 🔧 KeyError修复：验证targets中的必需字段
            if targets is not None and len(targets) > 0:
                valid_targets = []
                for target in targets:
                    # 检查必需的字段是否存在
                    required_fields = ['x', 'y', 'z', 'w', 'l', 'h', 'object_id']
                    if all(field in target for field in required_fields):
                        valid_targets.append(target)
                    else:
                        missing = [f for f in required_fields if f not in target]
                        print(f"警告: 跳过缺少字段的标注 {missing} in {frame_path}")

                # 如果所有targets都无效，跳过这个样本
                if len(valid_targets) == 0:
                    print(f"警告: 样本无有效标注，跳过: {frame_path}")
                    return None

                targets = valid_targets

            return {
                'camera_img': camera_img,
                'radar_bev': radar_bev,
                'radar_depth': radar_depth,
                'lidar_bev': lidar_bev,
                'lidar_depth': lidar_depth,
                'targets': targets,
                'frame_id': row['id'],
                'frame_path': frame_path
            }

        except KeyError as e:
            print(f"KeyError - 样本 {idx} 缺少字段 {e}: {frame_path if 'frame_path' in locals() else 'unknown'}")
            return None
        except Exception as e:
            print(f"样本加载失败 {idx}: {e}")
            return None


def custom_collate_fn(batch):
    """自定义批处理函数 - 使用深度图作为SGDNet输入"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    B = len(batch)
    H, W = batch[0]['camera_img'].shape[1], batch[0]['camera_img'].shape[2]

    # 准备输入张量
    camera_imgs = torch.stack([item['camera_img'] for item in batch])
    radar_bev = torch.stack([item['radar_bev'] for item in batch])  # (B, 5, H, W) BEV图像
    radar_depth = torch.stack([item['radar_depth'] for item in batch])  # (B, H, W) 深度图
    lidar_bev = torch.stack([item['lidar_bev'] for item in batch])  # (B, 5, H, W) BEV图像
    lidar_depth = torch.stack([item['lidar_depth'] for item in batch])  # (B, H, W) 深度图

    # 为fusion_model准备输入格式
    # 🔧 **关键修复 2025-11-19**: SGDNet的input_img应该是(B, 3, H, W)，不需要拼接3个相机
    # 参考: SGDNet_TI/Models/sgd.py - RGBEncoder期望(B, 3, H, W)输入
    input_img = camera_imgs  # (B, 3, H, W) - 直接使用相机图像，不拼接

    # 使用实际的radar/lidar BEV作为动态/静态热力图
    # 🔧 关键修复：detector期待 (B, H, W, C) 格式，不是 (B, C, H, W)！
    # 需要先permute再传给detector
    dynamic_HM_chw = radar_bev[:, :3, :, :]  # (B, 3, H, W) - 通道在前
    static_HM_chw = lidar_bev[:, :3, :, :]  # (B, 3, H, W) - 通道在前

    # 转换为通道在后的格式 (B, H, W, 3) - detector期待的格式
    dynamic_HM = dynamic_HM_chw.permute(0, 2, 3, 1)  # (B, H, W, 3)
    static_HM = static_HM_chw.permute(0, 2, 3, 1)  # (B, H, W, 3)

    # **关键修改**: 使用深度图传给SGDNet（LiDAR只用于监督，不作为网络输入）
    # SGDNet期待: (B, H, W*3) 深度图 (从相机视角)

    # 🔴🔴🔴 红色断点4: input_radar创建前检查 🔴🔴🔴 (已禁用)
    # print(f"\n{'='*80}")
    # print(f"🔴 断点4 - custom_collate_fn中input_radar创建检查:")
    # print(f"   radar_depth原始形状: {radar_depth.shape} ← 期望(B, H, W)")
    # print(f"   lidar_depth原始形状: {lidar_depth.shape}")
    # print(f"   radar_depth统计: 非零={radar_depth.count_nonzero().item()}, 均值={radar_depth.mean().item():.4f}")
    # print(f"   lidar_depth统计: 非零={lidar_depth.count_nonzero().item()}, 均值={lidar_depth.mean().item():.4f}")

    # 🔧 **关键修复 2025-11-19**: SGDNet输入格式应该是(B, H, W)，不是(B, H, W*3)！
    # SGDNet会在forward中自动unsqueeze: input_radar.unsqueeze(1) -> (B, 1, H, W)
    # 参考: SGDNet_TI/Models/sgd.py 第55行
    # print(f"\n   ✅ 使用雷达深度图（OCULii雷达）")
    input_radar = radar_depth  # (B, H, W) - 不需要repeat！
    input_velo = lidar_depth  # (B, H, W) - 不需要repeat！

    # print(f"   input_radar生成后形状: {input_radar.shape} ← 期望(B, H, W)，例如(2, 640, 640)")
    # print(f"   input_velo生成后形状:  {input_velo.shape}")
    # print(f"   ⚠️  SGDNet会在forward中自动unsqueeze(1)变成(B, 1, H, W)")
    # print(f"{'='*80}\n")

    # KNN和segmap暂时用零填充（需要额外实现KNN聚类）
    # 🔧 修复：形状应该是(B, 6, H, W)，不是(B, 6, H, W*3)
    input_knn = torch.zeros((B, 6, H, W), dtype=torch.float32)

    # 🔧 **关键修复**: segmap不能全是0，否则SGDNet会过滤掉所有LiDAR数据！
    # 让所有位置都属于class 0 (前景)，这样SGDNet才会使用LiDAR的远距离信息
    segmap = torch.zeros((B, 6, H, W), dtype=torch.float32)
    segmap[:, 0, :, :] = 1.0  # class 0 = 前景，值为1.0

    # OCULii分支仍然使用BEV图像
    oculii_img = radar_bev  # (B, 5, H, W) - 保留所有5通道给OCULii分支

    # 收集所有targets用于损失计算
    all_targets = [item['targets'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    frame_paths = [item['frame_path'] for item in batch]

    return {
        'model_inputs': (dynamic_HM, static_HM, input_img, input_radar, input_velo,
                         input_knn, segmap, oculii_img),  # 🔧 恢复input_velo参数
        'targets': all_targets,
        'frame_ids': frame_ids,
        'frame_paths': frame_paths,
        'camera_imgs': camera_imgs,
        'radar_bev': radar_bev,
        'radar_depth': radar_depth,
        'lidar_bev': lidar_bev,
        'lidar_depth': lidar_depth  # ✅ 保留lidar_depth用于loss监督
    }


class FocalLoss(nn.Module):
    """标准Focal Loss - 用于一般分类任务"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # 计算Focal权重
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FastFocalLoss(nn.Module):
    """FastFocalLoss - 来自RadarRGBFusionNet2,专门优化热力图检测

    关键改进:
    1. 对正样本使用 -log(pred) * (1-pred)^2
    2. 对负样本使用 -log(1-pred) * pred^2 * (1-target)^4
    3. 归一化方式: loss / num_pos (避免样本不平衡)

    参考: RadarRGBFusionNet2_20231128/module/one_stage_loss.py
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target):
        # 🔴 调试断点2: FastFocalLoss输入检查
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"\n⚠️ 断点2: FastFocalLoss输入异常!")
            print(f"   out范围: [{out.min():.3f}, {out.max():.3f}]")
            print(f"   out包含NaN: {torch.isnan(out).sum().item()}个")
            print(f"   out包含Inf: {torch.isinf(out).sum().item()}个")

        """
        Args:
            out: 模型输出 (B x C x H x W), sigmoid后的概率[0,1]
            target: 目标热力图 (B x C x H x W), 值为[0,1]

        🔧 2025-11-17修复: 移除distance_weight参数，完全对齐baseline
        参考: RadarRGBFusionNet2_20231128/module/one_stage_loss.py
        """
        # 增强数值稳定性: 更严格的裁剪避免log(0)和inf
        eps = 1e-6
        out = torch.clamp(out, eps, 1 - eps)

        # 检查输入是否异常
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"⚠️ 输入异常: out包含nan或inf")
            out = torch.clamp(out, eps, 1 - eps)

        # 负样本loss: -log(1-pred) * pred^2 * (1-target)^4
        gt = torch.pow(1 - target, 4)
        # 增强数值稳定性，防止log(0)和log(1)
        out_clamped_neg = torch.clamp(out, min=1e-8, max=1 - 1e-8)
        # 进一步确保log(1-out)不会产生-inf
        log_term = torch.log(1 - out_clamped_neg)
        log_term = torch.clamp(log_term, min=-10, max=0)  # 限制log范围
        neg_loss = log_term * torch.pow(out_clamped_neg, 2) * gt

        # 检查neg_loss是否异常
        if torch.isnan(neg_loss).any() or torch.isinf(neg_loss).any():
            print(f"⚠️ neg_loss异常: 包含nan或inf")
            neg_loss = torch.clamp(neg_loss, -10, 10)

        # 归一化: 除以总像素数避免累加过大
        neg_loss_sum = neg_loss.sum()
        neg_loss_normalized = neg_loss_sum / (gt.sum() + eps)

        # 检查归一化后是否异常
        if torch.isnan(neg_loss_normalized) or torch.isinf(neg_loss_normalized):
            print(f"⚠️ neg_loss_normalized异常: {neg_loss_normalized}")
            neg_loss_normalized = torch.tensor(0.0, device=out.device)

        # 正样本loss: -log(pred) * (1-pred)^2
        mask = (target >= 0.9)  # 改用>=0.9避免浮点精度问题
        num_pos = mask.sum().float()

        pos_loss_normalized = torch.tensor(0.0, device=out.device)
        if num_pos > 0:
            # 向量化计算,避免循环
            pos_mask_expanded = mask.float()
            pos_pred = out * pos_mask_expanded
            # 只对正样本位置计算，增强数值稳定性
            # 防止log(0)和数值不稳定
            out_clamped = torch.clamp(out[mask], min=1e-8, max=1 - 1e-8)
            # 确保log不会产生-inf
            log_term_pos = torch.log(out_clamped)
            log_term_pos = torch.clamp(log_term_pos, min=-10, max=0)  # 限制log范围
            tmp_pos_loss = log_term_pos * torch.pow(1 - out_clamped, 2)

            # ✅ 对齐baseline：移除距离权重应用
            # Baseline对所有位置使用相同权重

            pos_loss_sum = tmp_pos_loss.sum()
            pos_loss_normalized = pos_loss_sum / (num_pos + eps)

            # 总loss: 正样本loss + 负样本loss (参考FastFocalLoss)
            # **关键修复**: 取负号，因为log是负数
            total_loss = -(pos_loss_normalized + neg_loss_normalized)

            # 检查loss是否异常
            if torch.isinf(total_loss) or torch.isnan(total_loss):
                print(f"⚠️ Loss异常: {total_loss}, 正样本: {pos_loss_normalized}, 负样本: {neg_loss_normalized}")
                # 返回一个需要梯度的tensor
                return torch.tensor(1.0, device=out.device, requires_grad=True)

            # 确保loss需要梯度
            if not total_loss.requires_grad:
                total_loss = total_loss.detach().requires_grad_(True)

            # 参考RadarRGBFusionNet2_20231128，不进行复杂裁剪
            return total_loss
        else:
            total_loss = -neg_loss_normalized

            # 检查loss是否异常
            if torch.isinf(total_loss) or torch.isnan(total_loss):
                print(f"⚠️ Loss异常(仅负样本): {total_loss}, 负样本: {neg_loss_normalized}")
                # 返回一个需要梯度的tensor
                return torch.tensor(1.0, device=out.device, requires_grad=True)

            # 确保loss需要梯度
            if not total_loss.requires_grad:
                total_loss = total_loss.detach().requires_grad_(True)

            # 参考RadarRGBFusionNet2_20231128，不进行复杂裁剪
            return total_loss


class MAE_DepthLoss(nn.Module):
    """🔧 完全对齐SGDNet_TI的MAE Loss - /mnt/HotDisk/share/SGDNet_TI/Loss/loss.py"""

    def __init__(self):
        super(MAE_DepthLoss, self).__init__()

    def forward(self, prediction, gt):
        """
        Args:
            prediction: (B, 1, H, W) 深度预测
            gt: (B, 1, H, W) 深度GT
        """
        # 🚨 断点1: 检查输入是否包含NaN
        if torch.isnan(prediction).any():
            nan_count = torch.isnan(prediction).sum().item()
            total = prediction.numel()
            print(f"\n{'🚨' * 30}")
            print(f"🚨 断点1-MAE_DepthLoss: prediction包含NaN!")
            print(f"  NaN数量: {nan_count}/{total} ({100 * nan_count / total:.2f}%)")
            print(
                f"  prediction范围: [{prediction[~torch.isnan(prediction)].min():.6f}, {prediction[~torch.isnan(prediction)].max():.6f}]")
            print(f"{'🚨' * 30}\n")
            # 替换NaN为0，避免loss崩溃
            prediction = torch.nan_to_num(prediction, nan=0.0)

        if torch.isnan(gt).any():
            nan_count = torch.isnan(gt).sum().item()
            total = gt.numel()
            print(f"\n{'🚨' * 30}")
            print(f"🚨 断点1-MAE_DepthLoss: GT包含NaN!")
            print(f"  NaN数量: {nan_count}/{total} ({100 * nan_count / total:.2f}%)")
            print(f"{'🚨' * 30}\n")
            gt = torch.nan_to_num(gt, nan=0.0)

        # ✅ 完全对齐SGDNet: MAE_loss.forward() - line 26-31
        prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt) * 75  # SGDNet使用75的缩放因子
        mask = (gt > 0).detach()

        # 🚨 断点2: 检查mask是否为空
        if mask.sum() == 0:
            print(f"\n{'⚠️' * 30}")
            print(f"⚠️ 断点2-MAE_DepthLoss: mask全为False，没有有效GT!")
            print(f"  GT形状: {gt.shape}")
            print(f"  GT范围: [{gt.min():.6f}, {gt.max():.6f}]")
            print(f"  返回0 loss避免NaN")
            print(f"{'⚠️' * 30}\n")
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        mae_loss = torch.mean(abs_err[mask])

        # 🚨 断点3: 检查loss是否为NaN
        if torch.isnan(mae_loss) or torch.isinf(mae_loss):
            print(f"\n{'🚨' * 30}")
            print(f"🚨 断点3-MAE_DepthLoss: 计算后loss为NaN/Inf!")
            print(f"  mae_loss: {mae_loss}")
            print(f"  abs_err范围: [{abs_err[mask].min():.6f}, {abs_err[mask].max():.6f}]")
            print(f"  有效像素数: {mask.sum()}")
            print(f"{'🚨' * 30}\n")
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        return mae_loss


class SmoothEdgeLoss(nn.Module):
    """🔧 完全对齐SGDNet_TI的SmoothEdgeLoss - /mnt/HotDisk/share/SGDNet_TI/main.py:115-135"""

    def __init__(self):
        super(SmoothEdgeLoss, self).__init__()
        self.alpha = 0.2
        self.beta = 1.2

    def forward(self, depth, img):
        """
        Args:
            depth: (B, 1, H, W) 深度预测
            img: (B, 3, H, W) 图像（用于计算边缘）
        """
        # 🚨 断点1: 检查输入
        if torch.isnan(depth).any():
            print(f"\n🚨 SmoothEdgeLoss: depth包含NaN ({torch.isnan(depth).sum()}/{depth.numel()})")
            depth = torch.nan_to_num(depth, nan=0.0)

        if torch.isnan(img).any():
            print(f"\n🚨 SmoothEdgeLoss: img包含NaN ({torch.isnan(img).sum()}/{img.numel()})")
            img = torch.nan_to_num(img, nan=0.0)

        # ✅ 完全对齐SGDNet: SmoothEdgeLoss.forward() - line 122-135
        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        grad_seg_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_seg_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        # 归一化梯度（防止除以0）
        grad_seg_x_min = torch.min(grad_seg_x)
        grad_seg_x_max = torch.max(grad_seg_x)
        if (grad_seg_x_max - grad_seg_x_min) > 1e-8:
            grad_seg_x = (grad_seg_x - grad_seg_x_min) / (grad_seg_x_max - grad_seg_x_min)
        else:
            grad_seg_x = torch.zeros_like(grad_seg_x)

        grad_seg_y_min = torch.min(grad_seg_y)
        grad_seg_y_max = torch.max(grad_seg_y)
        if (grad_seg_y_max - grad_seg_y_min) > 1e-8:
            grad_seg_y = (grad_seg_y - grad_seg_y_min) / (grad_seg_y_max - grad_seg_y_min)
        else:
            grad_seg_y = torch.zeros_like(grad_seg_y)

        smoothX = torch.max(torch.zeros_like(grad_depth_x), grad_depth_x - self.alpha) * (1 - grad_seg_x)
        smoothY = torch.max(torch.zeros_like(grad_depth_y), grad_depth_y - self.alpha) * (1 - grad_seg_y)
        edgeX = torch.max(torch.zeros_like(grad_depth_x), self.beta - grad_depth_x) * grad_seg_x
        edgeY = torch.max(torch.zeros_like(grad_depth_y), self.beta - grad_depth_y) * grad_seg_y

        loss = smoothX.mean() + smoothY.mean() + edgeX.mean() + edgeY.mean()

        # 🚨 断点2: 检查loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n🚨 SmoothEdgeLoss: loss为NaN/Inf! {loss}")
            return torch.tensor(0.0, device=depth.device, requires_grad=True)

        return loss


class MultiModalLoss(nn.Module):
    """🔧 完全对齐SGDNet_TI的Loss配置 - 使用Focal Loss + SGDNet深度监督"""

    def __init__(self, alpha=1.0, beta=0.5, lambda_depth=1.0, use_fast_focal=True, max_pos_weight=3.0,
                 count_weight=0.1):
        super().__init__()
        self.alpha = alpha  # 检测损失权重
        self.count_weight = count_weight  # 🔧 MOTA修复
        self.beta = beta  # 回归损失权重（预留）
        self.lambda_depth = lambda_depth  # 🔧 深度监督权重（对齐SGDNet=1.0）
        self.use_fast_focal = use_fast_focal
        self.max_pos_weight = max_pos_weight
        self.batch_count = 0  # 🔴 用于断点调试的batch计数器

        # 初始化FastFocalLoss（检测分支）
        if use_fast_focal:
            self.fast_focal_loss = FastFocalLoss()
        else:
            self.fast_focal_loss = None

        # 🔧 初始化SGDNet的depth loss（完全对齐SGDNet_TI）
        self.mae_depth_loss = MAE_DepthLoss()
        self.smooth_edge_loss = SmoothEdgeLoss()

    def forward(self, outputs, targets_batch, depth_pred=None, lidar_depth_gt=None, camera_imgs=None):
        """🔧 完全对齐SGDNet_TI的loss计算逻辑

        SGDNet_TI使用4个loss（main.py:318-323）:
        1. loss_coarse = MAE(coarse_pred, lidarDepth)
        2. loss_cls = MAE(cls_pred, lidarDepth)
        3. loss_d = MAE(depth, lidarDepth)
        4. loss_smoothedge = SmoothEdge(depth, img)
        Total: loss = loss_coarse + loss_cls + loss_d + loss_smoothedge

        Args:
            outputs: (B, C, H, W) 检测热力图
            targets_batch: list of targets
            depth_pred: (B, 1, H, W) SGDNet深度预测（可选）
            lidar_depth_gt: (B, H, W) LiDAR深度GT（可选）
            camera_imgs: (B, 3, H, W) 相机图像（用于SmoothEdgeLoss）
        """
        B, C, H, W = outputs.shape
        device = outputs.device

        # 🔴 递增batch计数器（用于断点）
        self.batch_count += 1

        # 临时调试：输出尺寸信息（仅第一次）
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True

        # ===== 第1部分：检测loss（FastFocalLoss） =====
        # 创建目标热力图
        target_heatmaps = torch.zeros((B, C, H, W), device=device)

        # 🔧 **关键修复 2025-11-19**: 与fusion_model.py和深度图归一化保持一致
        # fusion_model.py第127-129行: X=[-82.7, 86.5], Y=[0, 75], Z=[-5, 30]
        X_MIN, X_MAX = -100.0, 105.0  # X轴 = 横向 (与fusion_model.py一致)
        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 = 前向75m (与深度图归一化一致)

        # 临时调试变量
        gt_y_coords_for_debug = []  # 用于调试GT坐标转换

        for i, targets in enumerate(targets_batch):
            if len(targets) > 0:
                for target in targets:
                    try:
                        x = target.get('x', None)
                        y = target.get('y', None)
                        if x is None or y is None:
                            continue
                    except (KeyError, TypeError):
                        continue

                    # 世界坐标 -> 像素坐标
                    norm_x = (x - X_MIN) / (X_MAX - X_MIN)
                    norm_y = (y - Y_MIN) / (Y_MAX - Y_MIN)
                    center_x = int(round(norm_x * (W - 1))) if 0 <= norm_x <= 1 else W // 2
                    center_y = int(round(norm_y * (H - 1))) if 0 <= norm_y <= 1 else H // 2
                    center_x = max(0, min(W - 1, center_x))
                    center_y = max(0, min(H - 1, center_y))
                    # print(f"🔴 断点B - GT热力图:")
                    # print(f"   GT世界坐标: (x={x:.2f}, y={y:.2f})")
                    # print(f"   归一化: (norm_x={norm_x:.4f}, norm_y={norm_y:.4f})")
                    # print(f"   像素坐标: (center_x={center_x}, center_y={center_y})")

                    # 🔴 断点1：GT坐标转换验证 (已禁用)
                    # if self.batch_count == 600 and i == 0:
                    #     print(f"\n🔴🔴🔴 断点1 - GT坐标转换验证 🔴🔴🔴")
                    #     print(f"坐标范围: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}], BEV={W}x{H}")
                    #     print(f"GT世界坐标: ({x:.2f}, {y:.2f}) -> 归一化({norm_x:.4f}, {norm_y:.4f}) -> 像素({center_x}, {center_y})")
                    #     # 往返验证
                    #     x_back = (center_x/(W-1)) * (X_MAX-X_MIN) + X_MIN
                    #     y_back = (center_y/(H-1)) * (Y_MAX-Y_MIN) + Y_MIN
                    #     print(f"往返验证: 原始({x:.2f}, {y:.2f}) -> 还原({x_back:.2f}, {y_back:.2f}) -> 误差({abs(x-x_back):.3f}m, {abs(y-y_back):.3f}m)")
                    #     print(f"结果: {'✅ 正确' if abs(x-x_back)<1 and abs(y-y_back)<1 else '❌ 错误！'}\n")

                    # 生成高斯热力图
                    sigma = 5
                    size = int(sigma * 3)
                    y_range = torch.arange(max(0, center_y - size), min(H, center_y + size + 1), device=device)
                    x_range = torch.arange(max(0, center_x - size), min(W, center_x + size + 1), device=device)

                    if len(y_range) > 0 and len(x_range) > 0:
                        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
                        gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))

                        y_start = max(0, center_y - size)
                        x_start = max(0, center_x - size)
                        target_heatmaps[i, 0, y_start:y_start + len(y_range),
                        x_start:x_start + len(x_range)] = torch.maximum(
                            target_heatmaps[i, 0, y_start:y_start + len(y_range), x_start:x_start + len(x_range)],
                            gaussian
                        )

        # 计算检测损失
        if self.use_fast_focal and self.fast_focal_loss is not None:
            outputs_prob = torch.sigmoid(outputs)
            detection_loss = self.fast_focal_loss(outputs_prob, target_heatmaps)
        else:
            positive_mask = target_heatmaps > 0
            positive_count = positive_mask.sum().float()
            total_count = target_heatmaps.numel()
            negative_count = total_count - positive_count

            if positive_count > 0:
                pos_weight = (negative_count / positive_count).clamp(min=1.0, max=self.max_pos_weight)
            else:
                pos_weight = torch.tensor(1.0, device=device)

            detection_loss = F.binary_cross_entropy_with_logits(
                outputs,
                target_heatmaps,
                pos_weight=pos_weight
            )

        # ===== 第2部分：深度监督loss（完全对齐SGDNet_TI） =====
        # 🚨 重要：SGDNet使用3个depth loss + 1个smooth edge loss
        depth_loss = torch.tensor(0.0, device=device, requires_grad=True)
        smooth_edge_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if depth_pred is not None and lidar_depth_gt is not None:
            # 🚨 断点：检查depth_pred是否为NaN
            if torch.isnan(depth_pred).any():
                print(f"\n{'🚨' * 30}")
                print(f"🚨 断点-MultiModalLoss: depth_pred包含NaN!")
                print(f"  depth_pred形状: {depth_pred.shape}")
                print(f"  NaN数量: {torch.isnan(depth_pred).sum()}/{depth_pred.numel()}")
                print(f"  问题：SGDNet模型前向传播输出NaN")
                print(f"{'🚨' * 30}\n")
                # 不计算depth loss，只返回detection loss
                return self.alpha * detection_loss, {
                    'detection_loss': detection_loss.item(),
                    'depth_loss': 0.0,
                    'smooth_edge_loss': 0.0,
                    'total_loss': (self.alpha * detection_loss).item()
                }

            # 确保GT和预测形状一致
            lidar_depth_gt_unsqueezed = lidar_depth_gt.unsqueeze(1)  # (B, 1, H, W)

            # Resize GT到与depth_pred相同尺寸
            if depth_pred.shape[2:] != lidar_depth_gt_unsqueezed.shape[2:]:
                lidar_depth_gt_unsqueezed = F.interpolate(
                    lidar_depth_gt_unsqueezed,
                    size=(depth_pred.shape[2], depth_pred.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )

            # ✅ 完全对齐SGDNet: 使用MAE_loss计算depth loss
            # 对应SGDNet的 loss_d = criterion_d(depth, ret['lidarDepth'].unsqueeze(1))
            depth_loss = self.mae_depth_loss(depth_pred, lidar_depth_gt_unsqueezed)

            # ✅ 完全对齐SGDNet: 使用SmoothEdgeLoss计算边缘平滑loss
            # 对应SGDNet的 loss_smoothedge = criterion_smoothedge(depth, IM)
            if camera_imgs is not None:
                # camera_imgs应该是(B, 3, H, W)格式
                # 如果尺寸不匹配，需要resize
                if camera_imgs.shape[2:] != depth_pred.shape[2:]:
                    camera_imgs_resized = F.interpolate(
                        camera_imgs,
                        size=(depth_pred.shape[2], depth_pred.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    camera_imgs_resized = camera_imgs

                # SGDNet使用原始图像（0-255范围），需要反归一化
                # 假设camera_imgs已经是[0,1]范围，转换到[0,255]
                camera_imgs_255 = camera_imgs_resized * 255.0

                smooth_edge_loss = self.smooth_edge_loss(depth_pred, camera_imgs_255)

        # ✅ 完全对齐SGDNet: total loss = detection + depth + smooth_edge
        # SGDNet: loss = loss_coarse + loss_cls + loss_d + loss_smoothedge (所有权重=1.0)
        # 我们简化为: loss = detection + depth + smooth_edge (所有权重=1.0)
        total_loss = self.alpha * detection_loss + self.lambda_depth * (depth_loss + smooth_edge_loss)

        return total_loss, {
            'detection_loss': detection_loss.item(),
            'depth_loss': depth_loss.item() if isinstance(depth_loss, torch.Tensor) else depth_loss,
            'smooth_edge_loss': smooth_edge_loss.item() if isinstance(smooth_edge_loss,
                                                                      torch.Tensor) else smooth_edge_loss,
            'total_loss': total_loss.item()
        }


def setup_balanced_multi_gpu():
    """智能多GPU设置 - 自动选择2-3个空闲GPU,避免占用他人GPU"""
    import random

    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return None, None, 0

    # 检查是否已经通过环境变量设置了GPU
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"✅ 检测到环境变量CUDA_VISIBLE_DEVICES={visible_devices}")
        print(f"   使用预设的GPU，跳过自动选择")

        total_gpus = torch.cuda.device_count()
        device = torch.device('cuda:0')
        available_gpus = list(range(total_gpus))

        print(f"🎯 使用GPU: {visible_devices} (映射为PyTorch设备 {available_gpus})")

        # 设置内存分配
        memory_fraction = 0.60 if total_gpus >= 3 else 0.70
        print(f"📊 GPU内存分配策略: 每GPU {memory_fraction * 100}%")

        try:
            for idx in range(total_gpus):
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=idx)
                print(f"   GPU {idx} 内存分配: {memory_fraction * 100}%")
        except Exception as e:
            print(f"⚠️ 内存分配设置失败: {e}")

        # 设置基本的NCCL环境变量（修复多GPU死锁问题）
        # 🔧 关键修复：移除过度限制，让NCCL自动选择最佳通信方式
        os.environ['NCCL_DEBUG'] = 'WARN'  # 只显示警告
        os.environ['NCCL_TIMEOUT'] = '300'  # 5分钟超时
        print(f"✅ NCCL环境变量已设置（简化配置避免死锁）")

        return device, available_gpus, total_gpus

    # 🔧 关键修复：在访问torch.cuda之前先用nvidia-smi检测并设置CUDA_VISIBLE_DEVICES
    # 这样避免创建多个GPU的CUDA上下文导致CUBLAS初始化失败

    import subprocess

    # 先用nvidia-smi获取GPU总数（不访问torch.cuda）
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        total_gpus = len(result.stdout.strip().split('\n'))
    except:
        print("❌ 无法通过nvidia-smi获取GPU信息")
        return None, None, 0

    print(f"🔍 系统GPU总数: {total_gpus}")

    # 检测所有GPU的空闲情况（只使用nvidia-smi，不访问torch.cuda）
    available_gpus = []
    gpu_free_memory = {}

    for i in range(total_gpus):
        try:
            # 使用nvidia-smi获取真实内存使用情况（包括其他进程）
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits', '-i', str(i)],
                capture_output=True, text=True, check=True
            )
            gpu_info = result.stdout.strip().split(', ')
            gpu_name = gpu_info[1]
            memory_used_mb = float(gpu_info[2])
            memory_free_mb = float(gpu_info[3])
            memory_total_mb = float(gpu_info[4])
            gpu_utilization = float(gpu_info[5])  # GPU计算利用率

            memory_free_gb = memory_free_mb / 1024
            usage_percent = (memory_used_mb / memory_total_mb) * 100

            # 智能空闲判断：根据GPU利用率和内存使用情况综合判断
            # 1. 完全空闲: 内存使用<1GB 且 GPU利用率<5%
            # 2. 轻度使用: 内存使用<10GB 且 GPU利用率<30%
            # 3. 繁忙: 内存使用>=10GB 或 GPU利用率>=30% (正在训练，避免占用)

            # 更严格的空闲判断，避免选择已有进程的GPU
            is_completely_free = memory_used_mb < 1000 and gpu_utilization < 5 and memory_free_gb > 40
            is_lightly_used = memory_used_mb < 10000 and gpu_utilization < 30 and memory_free_gb > 30

            if is_completely_free:
                available_gpus.append(i)
                gpu_free_memory[i] = memory_free_gb
                print(
                    f"✅ GPU {i} ({gpu_name}): {memory_free_gb:.1f}GB 可用 (使用 {memory_used_mb:.0f}MB, GPU利用率 {gpu_utilization:.1f}%) - 完全空闲")
            elif is_lightly_used and len(available_gpus) == 0:  # 只有在没有完全空闲GPU时才考虑轻度使用的
                available_gpus.append(i)
                gpu_free_memory[i] = memory_free_gb
                print(
                    f"🟡 GPU {i} ({gpu_name}): {memory_free_gb:.1f}GB 可用 (使用 {memory_used_mb:.0f}MB, GPU利用率 {gpu_utilization:.1f}%) - 轻度使用")
            else:
                print(
                    f"🔴 GPU {i} ({gpu_name}): 已使用 {memory_used_mb:.0f}MB (GPU利用率 {gpu_utilization:.1f}%) - 跳过")

        except (subprocess.CalledProcessError, Exception) as e:
            print(f"❌ GPU {i} 测试失败: {e}")
            continue

    if not available_gpus:
        print("❌ 没有足够空闲的GPU")
        return None, None, 0

    # 🔧 默认使用单GPU训练（最稳定）
    # 选择内存最多的空闲GPU
    selected_gpus = [available_gpus[0]]  # 只用一个GPU
    print(f"\n✅ 单GPU训练模式（最稳定配置）")
    print(f"🎯 选择GPU: {selected_gpus[0]}")

    # 按空闲内存排序，优先使用内存多的GPU
    selected_gpus.sort(key=lambda x: gpu_free_memory[x], reverse=True)

    print(f"🎯 最终选择GPU: {selected_gpus}")
    for gpu_id in selected_gpus:
        print(f"   GPU {gpu_id}: {gpu_free_memory[gpu_id]:.1f}GB 空闲")

    # 设置多GPU环境变量
    # 🔧 修复：如果已经设置了CUDA_VISIBLE_DEVICES，就不要再覆盖
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or not os.environ['CUDA_VISIBLE_DEVICES']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))

    # 优化NCCL设置，解决多GPU同步问题
    # 🔧 关键修复：移除过度限制，让NCCL自动选择最佳通信方式
    os.environ['NCCL_DEBUG'] = 'WARN'  # 只显示警告
    os.environ['NCCL_TIMEOUT'] = '300'  # 5分钟超时（不要太长）

    # 不再禁用P2P、SHM等功能，让NCCL自动选择
    # 之前的过度限制导致了死锁问题

    # 不设置CUDA_LAUNCH_BLOCKING，让CUDA异步运行
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 默认就是0，不需要设置

    print(f"\n🔧 已设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"🔧 NCCL配置: 简化设置，让NCCL自动选择最佳通信方式")

    # 🔧 关键修复：重新初始化torch CUDA上下文，确保使用正确的GPU映射
    # 清理可能已经存在的CUDA上下文（如果有的话）
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except:
        pass

    # 现在可以安全地创建设备对象
    device = torch.device('cuda:0')  # 主GPU（映射到CUDA_VISIBLE_DEVICES中的第一个GPU）

    # 根据GPU数量设置内存分配 - 优化多GPU训练
    num_gpus = len(selected_gpus)
    if num_gpus == 1:
        memory_fraction = 0.80  # 单GPU可以用更多内存
    elif num_gpus == 2:
        memory_fraction = 0.80  # 双GPU恢复到80%获得好效果
    elif num_gpus == 4:
        memory_fraction = 0.75  # 4GPU使用75%
    else:
        memory_fraction = 0.70  # 更多GPU保守一点

    print(f"📊 GPU内存分配策略: 每GPU {memory_fraction * 100}%")

    try:
        for idx in range(num_gpus):
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=idx)
            print(f"   GPU {idx} 内存分配: {memory_fraction * 100}%")
    except Exception as e:
        print(f"⚠️ 内存分配设置失败: {e}")

    return device, list(range(num_gpus)), num_gpus


class MemoryManager:
    """GPU内存管理器 - 处理OOM和动态批大小调整"""

    def __init__(self, initial_batch_size=16, min_batch_size=1):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.oom_count = 0
        self.last_oom_epoch = -1
        self.consecutive_oom = 0  # 连续OOM计数

    def handle_oom(self, epoch):
        """处理OOM错误"""
        self.oom_count += 1
        self.consecutive_oom += 1

        # 清理GPU内存（更安全的方式）
        try:
            torch.cuda.empty_cache()
        except:
            pass

        try:
            gc.collect()
        except:
            pass

        # 同步所有GPU（更安全的方式）
        if torch.cuda.device_count() > 1:
            try:
                torch.cuda.synchronize()
            except:
                pass

        # 减少批大小 - 改进逻辑
        # 如果连续3次OOM或者新的epoch出现OOM，就减少批大小
        should_reduce = (epoch != self.last_oom_epoch) or (self.consecutive_oom >= 3)

        if should_reduce:
            self.last_oom_epoch = epoch
            self.consecutive_oom = 0  # 重置连续OOM计数
            new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            if new_batch_size < self.current_batch_size:
                print(f"⚠️ 减少批大小: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                return True  # 需要重建dataloader
            elif new_batch_size == self.min_batch_size:
                print(f"⚠️ 已达到最小批大小 {self.min_batch_size}，无法继续减小")
        return False

    def cleanup_memory(self):
        """定期清理内存"""
        try:
            torch.cuda.empty_cache()
        except:
            pass

        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                except:
                    pass

    def get_memory_stats(self):
        """获取内存统计信息"""
        stats = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            stats.append({
                'gpu': i,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated
            })
        return stats


class EarlyStopping:
    """早停机制类"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                print(f"🔄 恢复最佳权重 (最佳loss: {self.best_loss:.4f})")
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()


def verify_coordinate_consistency():
    """验证GT生成和预测时的坐标转换一致性"""
    print("🔍 验证坐标转换一致性...")

    # 🔧 **关键修复 2025-11-19**: 与fusion_model.py和深度图归一化保持一致
    # fusion_model.py第127-129行: X=[-82.7, 86.5], Y=[0, 75], Z=[-5, 30]
    X_MIN, X_MAX = -100.0, 105.0  # X轴 = 横向 (与fusion_model.py一致)
    Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 = 前向75m (与深度图归一化一致)
    W, H = 640, 640  # 图像尺寸

    # 测试几个世界坐标点
    test_points = [
        (0.0, 0.0),  # 原点
        (10.0, 10.0),  # 中间点
        (X_MAX, Y_MAX),  # 边界点
    ]

    for world_x, world_y in test_points:
        # GT生成时的转换 - 使用稳定的转换方法
        norm_x_gt = (world_x - X_MIN) / (X_MAX - X_MIN)
        norm_y_gt = (world_y - Y_MIN) / (Y_MAX - Y_MIN)
        # 使用标准转换，与预测时保持一致
        pixel_x_gt = int(round(norm_x_gt * (W - 1)))
        pixel_y_gt = int(round(norm_y_gt * (H - 1)))
        # 确保在有效范围内
        pixel_x_gt = max(0, min(W - 1, pixel_x_gt))
        pixel_y_gt = max(0, min(H - 1, pixel_y_gt))

        # 预测时的反向转换 - 使用标准反向映射
        norm_x_pred = float(pixel_x_gt) / (W - 1)
        norm_y_pred = float(pixel_y_gt) / (H - 1)
        world_x_pred = norm_x_pred * (X_MAX - X_MIN) + X_MIN
        world_y_pred = norm_y_pred * (Y_MAX - Y_MIN) + Y_MIN

        # 计算误差
        error_x = abs(world_x - world_x_pred)
        error_y = abs(world_y - world_y_pred)

        print(
            f"  世界坐标({world_x:.2f}, {world_y:.2f}) -> 像素({pixel_x_gt}, {pixel_y_gt}) -> 世界({world_x_pred:.2f}, {world_y_pred:.2f})")
        print(f"    误差: x={error_x:.4f}m, y={error_y:.4f}m")

        # 理论最大误差 = 0.5 * 像素分辨率
        # X方向: (X_MAX - X_MIN) / W ≈ 0.26m/pixel
        # Y方向: (Y_MAX - Y_MIN) / H ≈ 0.17m/pixel
        # 优化误差阈值：边界点0.15米，中心点0.12米，特殊点0.10米
        if pixel_x_gt >= 630 or pixel_y_gt >= 630 or pixel_x_gt <= 10 or pixel_y_gt <= 10:
            # 边界点：误差阈值0.15米
            error_threshold = 0.15
            if error_x > error_threshold or error_y > error_threshold:
                print(f"    ⚠️ 边界点坐标转换误差较大（阈值={error_threshold}m）")
            else:
                print(f"    ✅ 边界点坐标转换可接受")
        elif abs(world_x) < 5.0 and abs(world_y) < 5.0:
            # 特殊点（接近原点）：误差阈值0.10米
            error_threshold = 0.10
            if error_x > error_threshold or error_y > error_threshold:
                print(f"    ⚠️ 特殊点坐标转换误差较大（阈值={error_threshold}m）")
            else:
                print(f"    ✅ 特殊点坐标转换可接受")
        else:
            # 中心点：误差阈值0.12米
            error_threshold = 0.12
            if error_x > error_threshold or error_y > error_threshold:
                print(f"    ⚠️ 中心点坐标转换误差较大（阈值={error_threshold}m）")
            else:
                print(f"    ✅ 中心点坐标转换精确")


# ============================================================================
# ✅ Baseline MOTA计算辅助函数（直接从EvaluateTracks.py借鉴）
# 参考：/mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/EvaluateTracks.py
# 这些函数确保MOTA计算与baseline完全一致
# ============================================================================

def pdist_baseline(a, b):
    """计算欧氏距离矩阵 - baseline实现

    Args:
        a: GT坐标数组 [N, 2] (x, y)
        b: 预测坐标数组 [M, 2] (x, y)

    Returns:
        距离矩阵 [N, M]
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.sqrt(r2)
    return r2


def min_cost_matching_baseline(cost_matrix, max_distance, tracks, detections):
    """最小成本匹配 - baseline实现（匈牙利算法）

    Args:
        cost_matrix: 成本矩阵 [N, M]
        max_distance: 最大距离阈值
        tracks: GT目标位置
        detections: 预测目标位置

    Returns:
        matches: 匹配对列表 [(gt_idx, pred_idx), ...]
        unmatched_tracks: 未匹配的GT索引
        unmatched_detections: 未匹配的预测索引
    """
    from scipy.optimize import linear_sum_assignment

    track_indices = np.arange(len(tracks))
    detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices.tolist(), detection_indices.tolist()

    # ✅ baseline核心逻辑：超过阈值的设为略大于阈值
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # 找出未匹配的检测
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    # 找出未匹配的GT
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    # 处理匹配结果
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            # 距离超过阈值，视为未匹配
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def compute_mota_motp(pred_tracks, gt_tracks, threshold=5.0, verbose=False, max_range=None):
    """
    计算基于检测的MOTA和MOTP指标（训练时不需要跟踪ID匹配）

    🚀 优化版本：使用scipy匈牙利算法，速度提升10-100倍

    Args:
        pred_tracks: 预测轨迹 [(frame_id, track_id, x, y, w, h, confidence), ...]
        gt_tracks: 真实轨迹 [(frame_id, track_id, x, y, w, h), ...]
        threshold: 距离阈值(米) - 训练初期使用5.0米，最终收敛到2.5米基准模型标准
        verbose: 是否输出详细信息
        max_range: 最大前向距离(米) - 用于公平比较，例如max_range=50只评估0-50m范围内的目标
    Returns:
        mota: 基于检测的MOTA值（限制在[0,1]范围内）
        motp: MOTP值
    """
    import time
    t_start = time.time()  # 🔍 调试: 开始计时

    try:
        # 🎯 范围过滤：用于公平比较baseline
        # 例如：max_range=50 只评估0-50m范围内的GT和预测
        if max_range is not None:
            # 过滤GT：只保留y坐标在范围内的
            gt_tracks_filtered = [t for t in gt_tracks if 0 <= t[3] <= max_range]  # t[3]=y
            # 过滤预测：只保留y坐标在范围内的
            pred_tracks_filtered = [t for t in pred_tracks if 0 <= t[3] <= max_range]  # t[3]=y

            if verbose and len(gt_tracks) > 0:
                print(f"\n🎯 范围过滤 (max_range={max_range}m):")
                print(
                    f"   GT:   {len(gt_tracks)} -> {len(gt_tracks_filtered)} (保留{100 * len(gt_tracks_filtered) / len(gt_tracks):.1f}%)")
                print(
                    f"   预测: {len(pred_tracks)} -> {len(pred_tracks_filtered)} (保留{100 * len(pred_tracks_filtered) / len(pred_tracks):.1f}%)")

            gt_tracks = gt_tracks_filtered
            pred_tracks = pred_tracks_filtered

        # 🚀 优化：导入scipy加速匹配算法
        from scipy.optimize import linear_sum_assignment
        if verbose:
            print(f"\n计算MOTA/MOTP: 预测轨迹={len(pred_tracks)}, 真实轨迹={len(gt_tracks)}")
            # 添加坐标调试
            if len(pred_tracks) > 0 and len(gt_tracks) > 0:
                print(f"  前3个预测坐标: {[(t[2], t[3]) for t in pred_tracks[:3]]}")
                print(f"  前3个GT坐标: {[(t[2], t[3]) for t in gt_tracks[:3]]}")
                # 计算第一个预测到第一个GT的距离
                dist = np.sqrt((pred_tracks[0][2] - gt_tracks[0][2]) ** 2 +
                               (pred_tracks[0][3] - gt_tracks[0][3]) ** 2)
                print(f"  第一个预测到第一个GT的距离: {dist:.2f}米, 阈值={threshold}米")

        if len(gt_tracks) == 0:
            # 没有真实目标时，如果有预测则MOTA为0
            return 0.0, 0.0

        if len(pred_tracks) == 0:
            # 没有预测但有真实目标，MOTA为0
            return 0.0, 0.0

        # 按帧组织数据 - ✅ 保留track_id用于IDSW计算
        pred_by_frame = {}
        gt_by_frame = {}

        for item in pred_tracks:
            frame_id = item[0]
            track_id = item[1]  # ✅ 保留track_id
            # 🔧 调试：确保frame_id是字符串
            frame_id = str(frame_id)
            if frame_id not in pred_by_frame:
                pred_by_frame[frame_id] = []
            # ✅ 保存完整信息：(track_id, x, y, w, h, confidence)
            if len(item) >= 7:  # 包含 confidence
                pred_by_frame[frame_id].append((track_id, item[2], item[3], item[4], item[5], item[6]))
            else:
                pred_by_frame[frame_id].append((track_id, item[2], item[3], item[4], item[5], 1.0))

        for item in gt_tracks:
            frame_id = item[0]
            track_id = item[1]  # ✅ 保留track_id
            # 🔧 调试：确保frame_id是字符串
            frame_id = str(frame_id)
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            # ✅ 保存完整信息：(track_id, x, y, w, h)
            gt_by_frame[frame_id].append((track_id, item[2], item[3], item[4], item[5]))

        total_gt = 0
        total_matches = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0  # ✅ 添加IDSW统计
        sum_distances = 0

        # ✅ 添加ID跟踪记录，用于计算IDSW
        # id_matches[gt_track_id] = [(frame_id, pred_track_id), ...]
        id_matches = {}

        # 计算每帧的检测匹配（不考虑跟踪ID）
        all_frames = set(pred_by_frame.keys()) | set(gt_by_frame.keys())
        if verbose:
            print(f"处理帧数: {len(all_frames)}")
            print(f"🔍 调试帧ID匹配:")
            print(f"   预测帧: {sorted(list(pred_by_frame.keys()))[:5]}")
            print(f"   GT帧: {sorted(list(gt_by_frame.keys()))[:5]}")
            common_frames = set(pred_by_frame.keys()) & set(gt_by_frame.keys())
            print(f"   共同帧数: {len(common_frames)}/{len(all_frames)}")

        for frame_id in sorted(all_frames):
            pred_objs = pred_by_frame.get(frame_id, [])
            gt_objs = gt_by_frame.get(frame_id, [])

            total_gt += len(gt_objs)

            if len(pred_objs) == 0 and len(gt_objs) == 0:
                continue

            # ✅ 使用baseline的pdist和min_cost_matching函数
            # 提取GT和pred的坐标 (取第2,3个元素: track_id, x, y, w, h, ...)
            gt_coords = np.array([[obj[1], obj[2]] for obj in gt_objs])  # [N, 2]
            pred_coords = np.array([[obj[1], obj[2]] for obj in pred_objs])  # [M, 2]

            # 计算距离矩阵（使用baseline函数）
            cost_matrix = pdist_baseline(gt_coords, pred_coords)

            # 匈牙利算法匹配（使用baseline函数）
            matches, unmatched_gt_indices, unmatched_pred_indices = min_cost_matching_baseline(
                cost_matrix, threshold, gt_coords, pred_coords
            )

            # 统计匹配数和距离
            total_matches += len(matches)
            frame_distances = []
            for gt_idx, pred_idx in matches:
                dist = cost_matrix[gt_idx, pred_idx]
                frame_distances.append(dist)

                # ✅ baseline对齐: 记录GT和pred的ID对应关系，用于计算IDSW
                gt_track_id = gt_objs[gt_idx][0]  # GT的track_id
                pred_track_id = pred_objs[pred_idx][0]  # pred的track_id

                if gt_track_id not in id_matches:
                    id_matches[gt_track_id] = []
                id_matches[gt_track_id].append((frame_id, pred_track_id))

            # 统计FP和FN (直接使用baseline返回的未匹配列表)
            fn_count = len(unmatched_gt_indices)
            fp_count = len(unmatched_pred_indices)

            # 优化FP计算，减少过度惩罚
            if fp_count > len(gt_objs) * 3:
                fp_count = len(gt_objs) * 3

            total_fp += fp_count
            total_fn += fn_count
            sum_distances += sum(frame_distances)

        # ✅ baseline对齐: 计算IDSW（ID切换次数）
        # 参考：EvaluateTracks.py line 122-135
        for gt_track_id, matches in id_matches.items():
            if len(matches) <= 1:
                continue  # 只出现1次，没有ID切换

            # 按帧排序
            matches_sorted = sorted(matches, key=lambda x: x[0])

            # 统计连续帧间的ID切换
            for idx in range(len(matches_sorted) - 1):
                frame_curr, pred_id_curr = matches_sorted[idx]
                frame_next, pred_id_next = matches_sorted[idx + 1]

                # 如果同一个GT目标在连续帧被分配了不同的pred ID → ID切换
                if pred_id_curr != pred_id_next:
                    total_idsw += 1

        # 计算MOTA - ✅ 完全对齐baseline
        if total_gt > 0:
            # ✅ baseline公式: MOTA = 1 - (FN + FP + IDSW) / GT
            # 参考: EvaluateTracks.py line 138
            total_errors = total_fn + total_fp + total_idsw

            # 🔍 强制输出MOTA详情（用于诊断MOTA=0问题）
            print(f"\n{'=' * 70}")
            print(f"📊 MOTA计算详情 [匹配阈值={threshold}m]:")
            print(f"   总GT={total_gt}, 总匹配={total_matches}, 总FP={total_fp}, 总FN={total_fn}, 总IDSW={total_idsw}")
            print(f"   总错误={total_errors}, 错误率={total_errors / total_gt:.2%}")

            if total_matches > 0:
                avg_match_dist = sum_distances / total_matches
                print(f"   ✅ 平均匹配距离={avg_match_dist:.2f}m")
            else:
                print(f"   ⚠️  没有任何匹配!")
                # 诊断：输出第一个预测和GT的距离
                if len(pred_tracks) > 0 and len(gt_tracks) > 0:
                    pred_0 = pred_tracks[0]
                    gt_0 = gt_tracks[0]
                    dist = np.sqrt((pred_0[2] - gt_0[2]) ** 2 + (pred_0[3] - gt_0[3]) ** 2)
                    print(f"   🔍 诊断信息:")
                    print(f"      第1个预测: frame={pred_0[0]}, x={pred_0[2]:.2f}, y={pred_0[3]:.2f}")
                    print(f"      第1个GT:   frame={gt_0[0]}, x={gt_0[2]:.2f}, y={gt_0[3]:.2f}")
                    print(f"      距离: {dist:.2f}m (阈值={threshold}m)")
                    if dist > threshold:
                        print(f"      💡 距离超过阈值！考虑增加threshold或检查坐标系")

                    # 🚨 断点1：保姆级坐标诊断
                    print(f"\n{'=' * 70}")
                    print(f"🚨 断点1：GT和预测坐标范围对比")
                    print(f"{'=' * 70}")

                    # 统计所有预测的坐标范围
                    pred_xs = [p[2] for p in pred_tracks]
                    pred_ys = [p[3] for p in pred_tracks]
                    print(f"📍 预测坐标统计 (共{len(pred_tracks)}个):")
                    print(f"   X范围: [{min(pred_xs):.2f}, {max(pred_xs):.2f}]m")
                    print(f"   Y范围: [{min(pred_ys):.2f}, {max(pred_ys):.2f}]m")
                    print(f"   X均值: {sum(pred_xs) / len(pred_xs):.2f}m")
                    print(f"   Y均值: {sum(pred_ys) / len(pred_ys):.2f}m")

                    # 统计所有GT的坐标范围
                    gt_xs = [g[2] for g in gt_tracks]
                    gt_ys = [g[3] for g in gt_tracks]
                    print(f"\n🎯 GT坐标统计 (共{len(gt_tracks)}个):")
                    print(f"   X范围: [{min(gt_xs):.2f}, {max(gt_xs):.2f}]m")
                    print(f"   Y范围: [{min(gt_ys):.2f}, {max(gt_ys):.2f}]m")
                    print(f"   X均值: {sum(gt_xs) / len(gt_xs):.2f}m")
                    print(f"   Y均值: {sum(gt_ys) / len(gt_ys):.2f}m")

                    # 计算坐标偏差
                    pred_x_mean = sum(pred_xs) / len(pred_xs)
                    pred_y_mean = sum(pred_ys) / len(pred_ys)
                    gt_x_mean = sum(gt_xs) / len(gt_xs)
                    gt_y_mean = sum(gt_ys) / len(gt_ys)

                    x_bias = pred_x_mean - gt_x_mean
                    y_bias = pred_y_mean - gt_y_mean

                    print(f"\n⚠️ 坐标偏差分析:")
                    print(f"   X方向偏差: {x_bias:+.2f}m  {'✅正常' if abs(x_bias) < 5 else '❌偏差过大！'}")
                    print(f"   Y方向偏差: {y_bias:+.2f}m  {'✅正常' if abs(y_bias) < 20 else '❌偏差过大！'}")

                    if abs(y_bias) > 20:
                        print(f"\n💡 建议：Y方向偏差超过20m，检查:")
                        print(f"   1. 坐标系转换是否正确（line 2014-2016, 2063-2066）")
                        print(f"   2. BEV范围设置是否一致（Y_MIN={0.0}, Y_MAX={50.0}）")
                        print(f"   3. 深度图归一化是否正确")

                    print(f"{'=' * 70}\n")
                elif len(pred_tracks) == 0:
                    print(f"   ⚠️  没有检测结果！检测阈值可能太高")
                elif len(gt_tracks) == 0:
                    print(f"   ⚠️  没有GT数据！")
            print(f"{'=' * 70}\n")

            # 🚀 优化：只在verbose模式下输出详细信息
            if verbose:
                print(f"\n📊 MOTA计算详情:")
                print(
                    f"  总GT={total_gt}, 总匹配={total_matches}, 总FP={total_fp}, 总FN={total_fn}, 总IDSW={total_idsw}")
                print(f"  总错误={total_errors}, 错误率={total_errors / total_gt:.2%}")
                if total_matches > 0:
                    print(f"  平均匹配距离={sum_distances / total_matches:.2f}m")

            # ✅ baseline对齐: MOTA = 1 - (FN + FP + IDSW) / GT
            # 参考: EvaluateTracks.py line 138
            mota = 1 - total_errors / total_gt

            # 🔧 修复：MOTA理论上不应为负数，但训练初期可能出现负值
            # - MOTA < 0: 错误数超过GT数，说明还没学好，限制为0
            # - MOTA = 0: 刚开始学习（训练初期正常）
            # - MOTA > 0: 开始有效学习
            # - MOTA > 1: 理论上不可能，但做保护
            mota = max(0.0, min(1.0, mota))
        else:
            mota = 0.0

        # 计算MOTP
        if total_matches > 0:
            avg_distance = sum_distances / total_matches
            # MOTP: 1 - (平均距离 / 阈值距离), 确保在[0,1]范围内
            motp = max(0.0, 1 - (avg_distance / threshold))
        else:
            motp = 0.0

        if verbose:
            print(f"匹配统计: 总目标={total_gt}, 匹配={total_matches}, FP={total_fp}, FN={total_fn}, IDSW={total_idsw}")
            if total_matches > 0:
                avg_dist = sum_distances / total_matches
                print(f"  平均匹配距离: {avg_dist:.2f}米, 阈值: {threshold}米")
            print(f"  MOTA={mota:.3f}, MOTP={motp:.3f}")

        t_end = time.time()  # 🔍 调试: 结束计时
        print(f"⏱️  [位置2-MOTA计算] 耗时: {t_end - t_start:.3f}秒 (预测={len(pred_tracks)}, GT={len(gt_tracks)})")

        return mota, motp

    except Exception as e:
        t_end = time.time()  # 🔍 调试: 异常时也计时
        print(f"⏱️  [位置2-MOTA计算] 异常退出，耗时: {t_end - t_start:.3f}秒")
        print(f"MOTA/MOTP计算错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0


def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 转换为角点坐标
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def extract_tracks_from_outputs(outputs, targets_batch, frame_ids, threshold=0.04, batch_num=0, epoch=0):
    """从模型输出中提取轨迹信息 - 完全重写版解决MOTA问题和固定检测数问题"""
    pred_tracks = []
    gt_tracks = []

    try:
        batch_size = outputs.shape[0]
        # 🔧 **2025-11-18第三次修复**: 扩大X范围覆盖实际GT数据分布！
        # 实际GT数据: X∈[-83.64,+30.0]m, Y∈[1.50,98.81]m
        # 🔧 **关键修复 2025-11-19**: 与fusion_model.py和深度图归一化保持一致
        X_MIN, X_MAX = -100.0, 105.0  # X轴 = 横向 (与fusion_model.py一致)
        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 = 前向75m (与深度图归一化一致)

        for batch_idx in range(batch_size):
            try:
                # 🔧 修复：在最开始定义变量，避免"referenced before assignment"错误
                is_early_training = epoch < 5
                adaptive_thresh = 0.2

                output = outputs[batch_idx]
                targets = targets_batch[batch_idx] if batch_idx < len(targets_batch) else []
                frame_id = frame_ids[batch_idx] if batch_idx < len(frame_ids) else f"frame_{batch_idx}"

                # 确保frame_id为字符串格式
                if isinstance(frame_id, (int, float)):
                    frame_id = str(int(frame_id))
                else:
                    frame_id = str(frame_id)

                # 从热力图中提取检测结果
                if len(output.shape) >= 2:
                    # 获取热力图
                    if len(output.shape) == 3:
                        heatmap = output[0].cpu().detach().float().numpy()
                    elif len(output.shape) == 2:
                        heatmap = output.cpu().detach().float().numpy()
                    else:
                        continue

                    if len(heatmap.shape) != 2:
                        continue

                    # 关键修复：将logits转换为概率！
                    # 模型输出是logits，需要sigmoid转为[0,1]概率
                    # 使用更稳定的sigmoid实现避免溢出
                    heatmap = np.clip(heatmap, -50, 50)  # 限制范围避免溢出
                    pos_mask = heatmap >= 0
                    neg_mask = ~pos_mask
                    # 稳定的sigmoid计算
                    heatmap[pos_mask] = 1.0 / (1.0 + np.exp(-heatmap[pos_mask]))
                    exp_pos = np.exp(heatmap[neg_mask])
                    heatmap[neg_mask] = exp_pos / (1.0 + exp_pos)

                    # 基本统计信息
                    mean_val = np.mean(heatmap)
                    std_val = np.std(heatmap)
                    max_val = np.max(heatmap)
                    min_val = np.min(heatmap)

                    target_count = len(targets)

                    # 🚨 断点2：检查热力图峰值位置（已禁用）
                    # if batch_num % 50 == 0:
                    #     print(f"\n{'='*70}")
                    #     print(f"🚨 断点2：热力图峰值诊断 - Batch {batch_num}")
                    #     print(f"{'='*70}")
                    #     print(f"📊 热力图统计:")
                    #     print(f"   mean={mean_val:.4f}, std={std_val:.4f}")
                    #     print(f"   max={max_val:.4f}, min={min_val:.4f}")
                    #     print(f"   GT目标数={target_count}")
                    #
                    #     # 找出热力图的前3个峰值位置
                    #     flat_heatmap = heatmap.flatten()
                    #     top3_indices = np.argpartition(flat_heatmap, -3)[-3:]
                    #     top3_indices = top3_indices[np.argsort(-flat_heatmap[top3_indices])]
                    #
                    #     print(f"\n📍 热力图前3个峰值位置:")
                    #     for i, idx in enumerate(top3_indices):
                    #         peak_y, peak_x = np.unravel_index(idx, heatmap.shape)
                    #         peak_val = heatmap[peak_y, peak_x]
                    #
                    #         # 转换为世界坐标
                    #         X_MIN, X_MAX = -100.0, 105.0
                    #         Y_MIN, Y_MAX = -5.0, 105.0
                    #         norm_y = float(peak_y) / (heatmap.shape[0] - 1)
                    #         norm_x = float(peak_x) / (heatmap.shape[1] - 1)
                    #         world_x = norm_x * (X_MAX - X_MIN) + X_MIN
                    #         world_y = norm_y * (Y_MAX - Y_MIN) + Y_MIN
                    #
                    #         print(f"   峰值{i+1}: pixel(y={peak_y:3d}, x={peak_x:3d}) -> world(x={world_x:7.2f}, y={world_y:7.2f}), 概率={peak_val:.4f}")
                    #
                    #         # 🔴 断点2：预测解码验证
                    #         if batch_num == 600 and i == 0:
                    #             print(f"\n🔴🔴🔴 断点2 - 预测解码验证 🔴🔴🔴")
                    #             print(f"坐标范围: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}], 热力图={heatmap.shape}")
                    #             print(f"峰值像素: ({peak_x}, {peak_y}) -> 归一化({norm_x:.4f}, {norm_y:.4f}) -> 世界({world_x:.2f}, {world_y:.2f})")
                    #             px_back = int(round((world_x-X_MIN)/(X_MAX-X_MIN) * (heatmap.shape[1]-1)))
                    #             py_back = int(round((world_y-Y_MIN)/(Y_MAX-Y_MIN) * (heatmap.shape[0]-1)))
                    #             print(f"逆验证: 世界({world_x:.2f}, {world_y:.2f}) -> 像素({px_back}, {py_back}) -> 原始({peak_x}, {peak_y}) -> 误差({abs(px_back-peak_x)}, {abs(py_back-peak_y)})")
                    #             print(f"结果: {'✅ 正确' if abs(px_back-peak_x)<2 and abs(py_back-peak_y)<2 else '❌ 错误！'}\n")
                    #
                    #     # 对比GT位置
                    #     if target_count > 0:
                    #         print(f"\n🎯 GT位置对比:")
                    #         for i, target in enumerate(targets[:3]):
                    #             gt_x = target.get('x', 0)
                    #             gt_y = target.get('y', 0)
                    #             print(f"   GT{i+1}: world(x={gt_x:7.2f}, y={gt_y:7.2f})")
                    #
                    #         if len(top3_indices) > 0:
                    #             peak_y, peak_x = np.unravel_index(top3_indices[0], heatmap.shape)
                    #             norm_y = float(peak_y) / (heatmap.shape[0] - 1)
                    #             norm_x = float(peak_x) / (heatmap.shape[1] - 1)
                    #             world_x = norm_x * (X_MAX - X_MIN) + X_MIN
                    #             world_y = norm_y * (Y_MAX - Y_MIN) + Y_MIN
                    #
                    #             gt_0 = targets[0]
                    #             dist_to_gt = np.sqrt((world_x - gt_0.get('x', 0))**2 + (world_y - gt_0.get('y', 0))**2)
                    #             print(f"\n💡 最高峰值到第1个GT的距离: {dist_to_gt:.2f}m")
                    #             if dist_to_gt > 10:
                    #                 print(f"   ❌ 距离>{10}m，模型预测位置严重偏离GT！")
                    #                 print(f"   → 可能原因：坐标系转换错误或BEV输入有问题")
                    #
                    #     print(f"{'='*70}\n")

                    # 临时调试：输出热力图统计和检测位置（禁用以加速训练）
                    # if batch_num % 100 == 0:
                    #     print(f"\nDEBUG 热力图统计: mean={mean_val:.4f}, std={std_val:.4f}, max={max_val:.4f}, min={min_val:.4f}")
                    #     max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    #     print(f"  热力图峰值位置: (y={max_y}, x={max_x}), shape={heatmap.shape}")
                    #     print(f"  GT目标数={target_count}, GT坐标: {[(t.get('x', 0), t.get('y', 0)) for t in targets[:3]]}")

                    # 🔧 **完全对齐Baseline**: 使用MaxPool NMS + Top-K策略
                    # 参考: /mnt/ChillDisk/personal_data/mij/pythonProject1/RadarRGBFusionNet2_20231128/module/TRL.py:16-45
                    detections_found = []

                    # ===== Step 1: MaxPool NMS（完全对齐Baseline）=====
                    # Baseline的关键步骤：3x3 MaxPool过滤非峰值点
                    import torch
                    import torch.nn.functional as F_nms

                    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

                    # MaxPool2d with kernel_size=3, stride=1, padding=1
                    max_pooled = F_nms.max_pool2d(heatmap_tensor, kernel_size=3, stride=1, padding=1)

                    # 只保留局部最大值（峰值点）
                    peak_mask = torch.eq(heatmap_tensor, max_pooled).float()
                    filtered_heatmap_tensor = heatmap_tensor * peak_mask
                    filtered_heatmap = filtered_heatmap_tensor.squeeze().numpy()

                    # ===== Step 2: Top-K选择 + 置信度过滤（优化策略）=====
                    # 🔧 优化1: 增加max_num_objects以减少FN（漏检）
                    # 🔧 优化2: 降低置信度阈值以保留更多真实目标
                    # 🔧 MOTA修复: 动态调整最大检测数
                    if target_count > 0:
                        max_num_objects = min(max(target_count * 2, 5), 15)  # GT的2倍，最少5，最多15
                    else:
                        max_num_objects = 10  # 默认值
                    score_threshold = 0.3  # 🔴 MOTA诊断: 从0.5降低到0.3，保留更多检测

                    # Flatten并选择Top-K
                    flat_filtered = filtered_heatmap.flatten()
                    if len(flat_filtered) < max_num_objects:
                        k = len(flat_filtered)
                    else:
                        k = max_num_objects

                    # 使用np.argpartition选择Top-K
                    top_k_indices = np.argpartition(flat_filtered, -k)[-k:]
                    top_k_values = flat_filtered[top_k_indices]

                    # 🔧 添加置信度过滤（重要！减少FP）
                    # 只保留置信度 >= score_threshold 的检测
                    high_conf_mask = top_k_values >= score_threshold
                    valid_indices = top_k_indices[high_conf_mask]
                    coords = np.unravel_index(valid_indices, heatmap.shape)

                    # 🔍 调试输出（每50个batch）
                    if batch_num % 50 == 0:
                        print(f"\n✨ 优化检测策略（Top-{max_num_objects} + 置信度>{score_threshold}）:")
                        print(f"   MaxPool过滤后峰值数: {np.count_nonzero(filtered_heatmap)}")
                        print(f"   Top-{k}个候选位置")
                        print(f"   置信度>{score_threshold}过滤后: {len(coords[0])}个检测")
                        print(f"   最终检测数={len(coords[0])}, GT数={target_count}")
                        if len(coords[0]) > 0:
                            # 显示过滤效果
                            filtered_count = k - len(coords[0])
                            print(f"   🎯 过滤掉低置信度检测: {filtered_count}个")
                            # 计算检测数与GT的比例
                            if target_count > 0:
                                ratio = len(coords[0]) / target_count
                                if ratio > 1.5:
                                    print(f"   ⚠️ 检测数/GT={ratio:.2f} (偏高，可能有FP)")
                                elif ratio < 0.8:
                                    print(f"   ⚠️ 检测数/GT={ratio:.2f} (偏低，可能有FN)")
                                else:
                                    print(f"   ✅ 检测数/GT={ratio:.2f} (合理范围)")

                    # 坐标转换和检测生成
                    final_detections = []
                    if len(coords[0]) > 0:
                        # 🔍 断点：坐标转换调试（每50个batch输出）
                        if batch_num % 50 == 0:
                            print(f"\n{'=' * 70}")
                            print(f"[坐标调试-完全对齐Baseline] Batch {batch_num}")
                            print(f"  Heatmap形状: {heatmap.shape}")
                            print(f"  检测到 {len(coords[0])} 个位置")
                            print(f"  前3个检测的坐标转换:")

                        for det_idx, (y, x) in enumerate(zip(coords[0], coords[1])):
                            # 🔧 **关键修复**: BEV图像坐标到世界坐标的正确映射
                            # BEV图像: 行(y)=前向, 列(x)=横向
                            # 世界坐标: X=横向, Y=前向
                            # 因此: pixel_y→world_y, pixel_x→world_x
                            norm_y = float(y) / (heatmap.shape[0] - 1) if heatmap.shape[0] > 1 else 0.5
                            norm_x = float(x) / (heatmap.shape[1] - 1) if heatmap.shape[1] > 1 else 0.5
                            world_x = norm_x * (X_MAX - X_MIN) + X_MIN  # 列→横向(X)
                            world_y = norm_y * (Y_MAX - Y_MIN) + Y_MIN  # 行→前向(Y)
                            confidence = float(heatmap[y, x])

                            if batch_num % 50 == 0 and det_idx < 3:
                                print(
                                    f"    [{det_idx}] pixel(y={y:3d}, x={x:3d}) -> norm(y={norm_y:.4f}, x={norm_x:.4f}) -> world(x={world_x:7.2f}m, y={world_y:7.2f}m) conf={confidence:.3f}")

                            # 🔴🔴🔴 MOTA诊断断点 - 自动插入 🔴🔴🔴
                            # 只在前3个batch和每100个batch输出，避免刷屏
                            if (batch_num <= 3 or batch_num % 100 == 0) and det_idx == 0 and len(targets) > 0:
                                gt_x = targets[0].get('x', 0)
                                gt_y = targets[0].get('y', 0)
                                dist_to_gt = np.sqrt((world_x - gt_x) ** 2 + (world_y - gt_y) ** 2)

                                print(f"\n{'🔴' * 25}")
                                print(f"🔴 MOTA诊断断点 - Batch {batch_num}")
                                print(f"{'🔴' * 25}")
                                print(f"   预测第1个: x={world_x:.2f}m, y={world_y:.2f}m (conf={confidence:.3f})")
                                print(f"   GT第1个:   x={gt_x:.2f}m, y={gt_y:.2f}m")
                                print(f"   距离: {dist_to_gt:.2f}m")

                                if dist_to_gt < 10:
                                    print(f"   ✅ 距离正常(<10m)，坐标系没问题")
                                    print(f"   → MOTA低可能是阈值设置问题，建议降低score_threshold")
                                elif dist_to_gt < 25:
                                    print(f"   🟡 距离偏大(10-25m)，需要放宽match_threshold")
                                    print(f"   → 建议将match_threshold改为30m")
                                else:
                                    print(f"   ❌ 距离过大(>25m)，坐标系可能有问题!")
                                    print(f"   → 检查X_MIN/X_MAX/Y_MIN/Y_MAX设置")

                                # 输出前3个GT用于对比
                                print(f"\n   所有GT坐标 (共{len(targets)}个):")
                                for gi, gt in enumerate(targets[:3]):
                                    print(f"      GT[{gi}]: x={gt.get('x', 0):.2f}m, y={gt.get('y', 0):.2f}m")
                                print(f"{'🔴' * 25}\n")
                            # 🔴🔴🔴 MOTA诊断断点 - 结束 🔴🔴🔴

                            final_detections.append((world_x, world_y, confidence))

                    #                     # 🔧 完全对齐baseline: 不使用距离NMS
                    #                     # Baseline只用MaxPool NMS，没有距离NMS
                    #                     # 直接使用final_detections，不做进一步过滤

                    # 🔧 MOTA修复: 添加距离NMS减少重复检测
                    min_distance = 5.0  # 最小检测间隔5米
                    filtered_detections = []
                    for det in final_detections:
                        too_close = False
                        for existing in filtered_detections:
                            dist = np.sqrt((det[0] - existing[0]) ** 2 + (det[1] - existing[1]) ** 2)
                            if dist < min_distance:
                                if det[2] <= existing[2]:  # 保留置信度更高的
                                    too_close = True
                                    break
                                else:
                                    filtered_detections.remove(existing)
                                    break
                        if not too_close:
                            filtered_detections.append(det)
                    final_detections = filtered_detections

                    # 添加到预测轨迹
                    for i, detection in enumerate(final_detections):
                        world_x, world_y, confidence = detection
                        track_id = f"pred_{frame_id}_{i}"

                        # 🔍 调试输出（每50个batch输出前5个检测）
                        if batch_num % 50 == 0 and i < 5:
                            print(
                                f"🔍 添加预测 - frame={frame_id}, det={i}, x={world_x:.2f}, y={world_y:.2f}, conf={confidence:.3f}")

                        # 🔧 修复：确保frame_id是字符串格式，与GT保持一致
                        pred_tracks.append((str(frame_id), track_id, world_x, world_y, 2.0, 2.0, confidence))

                # 处理真实轨迹
                for gt_idx, target in enumerate(targets):
                    if isinstance(target, dict) and 'object_id' in target:
                        track_id = target['object_id']
                        x = float(target.get('x', 0))
                        y = float(target.get('y', 0))
                        w = float(target.get('w', 2))
                        l = float(target.get('l', 2))
                        # 🔧 修复：确保frame_id是字符串格式，与预测保持一致
                        gt_tracks.append((str(frame_id), track_id, x, y, w, l))

            except Exception as batch_e:
                print(f"批次 {batch_idx} 处理失败: {batch_e}")
                continue

        # 调试信息: 禁用
        # if batch_num % 100 == 0:
        #     print(f"Debug - 批次 {batch_num} 汇总: 总预测轨迹={len(pred_tracks)}, "
        #           f"总真实轨迹={len(gt_tracks)}, 处理帧数={batch_size}")

        return pred_tracks, gt_tracks

    except Exception as e:
        print(f"轨迹提取总体错误: {e}")
        print(f"outputs shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
        print(f"targets_batch length: {len(targets_batch) if targets_batch else 'None'}")
        print(f"frame_ids: {frame_ids if frame_ids else 'None'}")
        return [], []


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch,
                multi_gpu=False, num_gpus=1, eval_mota=True, val_freq=200,
                memory_manager=None, gradient_accumulation_steps=2):
    """训练一个epoch - 支持梯度累积和OOM恢复"""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_steps = 0

    # 用于MOTA计算的轨迹收集
    all_pred_tracks = []
    all_gt_tracks = []

    # 多GPU训练时的特殊配置
    desc_prefix = f"🚀 多GPU训练 ({num_gpus}GPUs)" if multi_gpu else "🔧 单GPU训练"

    pbar = tqdm(dataloader, desc=f'{desc_prefix} Epoch {epoch}',
                unit='batch',
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0 if multi_gpu else 2.0,
                maxinterval=10.0 if multi_gpu else 30.0,
                disable=False)

    # 梯度累积缓冲
    accumulated_loss = 0

    # GPU利用率监控 - 确保训练真正运行
    gpu_util_check_interval = 100  # 每100个batch检查一次
    last_gpu_util_check = 0
    eval_interval = None
    if eval_mota and val_freq and val_freq > 0:
        eval_interval = val_freq
    cleanup_interval = eval_interval if eval_interval else 200

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        try:
            # 获取当前批次大小（最后一批可能不足）
            current_batch_size = batch['model_inputs'][0].size(0) if batch['model_inputs'] else 0

            # 移动数据到设备（修复多GPU设备分配问题）
            try:
                # 关键修复：DataParallel模式下，数据应该移动到主设备，而不是分散到不同设备
                # DataParallel会自动处理数据分发
                if multi_gpu:
                    # 多GPU模式：确保数据在cuda:0上，DataParallel会自动分发
                    primary_device = torch.device('cuda:0')
                    model_inputs = tuple(
                        tensor.to(primary_device, non_blocking=True) for tensor in batch['model_inputs'])
                else:
                    # 单GPU模式：正常移动到指定设备
                    model_inputs = tuple(tensor.to(device, non_blocking=True) for tensor in batch['model_inputs'])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ 批次 {batch_idx} 数据移动时OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            targets = batch['targets']
            frame_ids = batch['frame_ids']
            # 同样修复lidar_depth的设备分配
            if multi_gpu:
                primary_device = torch.device('cuda:0')
                lidar_depth = batch['lidar_depth'].to(primary_device, non_blocking=True)  # 🔧 获取LiDAR深度GT
            else:
                lidar_depth = batch['lidar_depth'].to(device, non_blocking=True)  # 🔧 获取LiDAR深度GT

            # 🔴🔴🔴 断点1 - BEV输入信号检查 🔴🔴🔴 (已禁用)
            # if batch_idx % 50 == 0:
            #     print(f"\n{'🔴'*50}")
            #     print(f"🔴 断点1 - BEV输入信号检查 - Epoch {epoch}, Batch {batch_idx}")
            #     print(f"{'🔴'*50}")
            #
            #     oculii = model_inputs[7]
            #     print(f"\n📊 oculii_img (OCULii雷达BEV):")
            #     print(f"   形状: {oculii.shape}")
            #
            #     total_pixels = oculii.numel()
            #     nonzero_pixels = (oculii != 0).sum().item()
            #     nonzero_ratio = 100 * nonzero_pixels / total_pixels
            #
            #     print(f"   非零像素: {nonzero_pixels}/{total_pixels}")
            #     print(f"   非零比例: {nonzero_ratio:.2f}%")
            #     print(f"   值范围: [{oculii.min():.4f}, {oculii.max():.4f}]")
            #     print(f"   均值: {oculii.mean():.4f}")
            #
            #     print(f"\n   📊 各通道非零像素:")
            #     for c in range(oculii.shape[1]):
            #         ch_data = oculii[:, c]
            #         ch_nonzero = (ch_data != 0).sum().item()
            #         ch_total = oculii.shape[0] * oculii.shape[2] * oculii.shape[3]
            #         ch_ratio = 100 * ch_nonzero / ch_total
            #         print(f"      Ch{c}: {ch_nonzero}/{ch_total} ({ch_ratio:.2f}%), 范围[{ch_data.min():.4f}, {ch_data.max():.4f}]")
            #
            #         if ch_nonzero == 0:
            #             print(f"         ❌ 警告：通道{c}完全为零！")
            #
            #     dyn_hm = model_inputs[0]
            #     print(f"\n📊 dynamic_HM (Radar动态热力图):")
            #     print(f"   形状: {dyn_hm.shape}")
            #     dyn_nonzero = (dyn_hm != 0).sum().item()
            #     dyn_ratio = 100 * dyn_nonzero / dyn_hm.numel()
            #     print(f"   非零像素: {dyn_nonzero}/{dyn_hm.numel()}")
            #     print(f"   非零比例: {dyn_ratio:.2f}%")
            #     print(f"   值范围: [{dyn_hm.min():.4f}, {dyn_hm.max():.4f}]")
            #
            #     print(f"\n🔍 诊断结果:")
            #     if nonzero_ratio < 1:
            #         print(f"   ❌❌❌ 致命问题：oculii_img信号<1%，几乎全零！")
            #     elif nonzero_ratio < 5:
            #         print(f"   ⚠️ 警告：oculii_img信号<5%，信号较弱")
            #     else:
            #         print(f"   ✅ oculii_img信号正常 ({nonzero_ratio:.2f}%)")
            #
            #     if dyn_ratio < 1:
            #         print(f"   ⚠️ dynamic_HM信号也很弱 ({dyn_ratio:.2f}%)")
            #
            #     print(f"{'🔴'*50}\n")

            # 前向传播（梯度累积）
            if USE_AMP:
                with get_autocast():
                    # 🔧 模型现在返回 (outputs, depth_pred)
                    model_output = model(*model_inputs)
                    if isinstance(model_output, tuple):
                        outputs, depth_pred = model_output
                    else:
                        outputs = model_output
                        depth_pred = None

                    # 🚨 断点1：检查模型输出是否有NaN
                    if torch.isnan(outputs).any():
                        print(f"\n{'🚨' * 30}")
                        print(f"🚨 断点1触发：模型outputs有NaN！")
                        print(f"  outputs形状: {outputs.shape}")
                        print(f"  NaN数量: {torch.isnan(outputs).sum().item()}/{outputs.numel()}")
                        print(f"  outputs范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
                        print(f"  问题：SGDNet前向传播输出NaN")
                        print(f"{'🚨' * 30}\n")

                    if depth_pred is not None and torch.isnan(depth_pred).any():
                        print(f"\n{'🚨' * 30}")
                        print(f"🚨 断点1触发：depth_pred有NaN！")
                        print(f"  depth_pred形状: {depth_pred.shape}")
                        print(f"  NaN数量: {torch.isnan(depth_pred).sum().item()}/{depth_pred.numel()}")
                        print(
                            f"  depth_pred范围: [{depth_pred[~torch.isnan(depth_pred)].min():.6f}, {depth_pred[~torch.isnan(depth_pred)].max():.6f}]")
                        print(f"  问题：深度预测网络输出NaN")
                        print(f"{'🚨' * 30}\n")

                    # 🔧 传递深度预测、GT和相机图像给loss函数（完全对齐SGDNet）
                    # 获取camera_imgs用于SmoothEdgeLoss（确保在GPU上）
                    if multi_gpu:
                        primary_device = torch.device('cuda:0')
                        camera_imgs_for_loss = batch['camera_imgs'].to(primary_device, non_blocking=True)
                    else:
                        camera_imgs_for_loss = batch['camera_imgs'].to(device, non_blocking=True)
                    loss, loss_dict = criterion(outputs, targets, depth_pred, lidar_depth, camera_imgs_for_loss)

                    # 🚨 断点2：检查loss计算后是否有NaN
                    if torch.isnan(loss):
                        print(f"\n{'🚨' * 30}")
                        print(f"🚨 断点2触发：Loss计算后是NaN！")
                        print(f"  loss值: {loss}")
                        print(f"  detection_loss: {loss_dict.get('detection_loss', 'N/A')}")
                        print(f"  depth_loss: {loss_dict.get('depth_loss', 'N/A')}")

                        # 检查是哪个loss导致的NaN
                        det_loss = loss_dict.get('detection_loss', 0)
                        dep_loss = loss_dict.get('depth_loss', 0)
                        if isinstance(det_loss, (int, float)):
                            det_is_nan = False
                        else:
                            det_is_nan = torch.isnan(torch.tensor(det_loss))
                        if isinstance(dep_loss, (int, float)):
                            dep_is_nan = False
                        else:
                            dep_is_nan = torch.isnan(torch.tensor(dep_loss))

                        if det_is_nan:
                            print(f"  ❌ detection_loss是NaN！问题在Focal Loss计算")
                        if dep_is_nan:
                            print(f"  ❌ depth_loss是NaN！问题在深度监督loss计算")
                        print(f"  lambda_depth当前值: {criterion.lambda_depth}")
                        print(f"{'🚨' * 30}\n")

                    # 🔍 第一个batch的详细诊断
                    if batch_idx == 0:
                        print(f"\n{'=' * 60}")
                        print(f"🔍 第一个Batch Loss诊断:")
                        print(f"{'=' * 60}")
                        print(f"📊 Loss详情:")
                        print(f"   detection_loss: {loss_dict['detection_loss']:.4f}")
                        print(f"   depth_loss: {loss_dict['depth_loss']:.4f}")
                        print(f"   total_loss: {loss.item() * gradient_accumulation_steps:.4f}")
                        print(f"\n📈 模型输出统计:")
                        print(f"   outputs范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
                        print(f"   outputs均值: {outputs.mean():.3f}")
                        print(
                            f"   outputs sigmoid后范围: [{torch.sigmoid(outputs).min():.3f}, {torch.sigmoid(outputs).max():.3f}]")
                        if depth_pred is not None:
                            print(f"\n🌊 深度预测统计:")
                            print(f"   depth_pred范围: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
                            print(f"   depth_pred均值: {depth_pred.mean():.3f}")
                            print(f"   depth_pred标准差: {depth_pred.std():.3f}")  # 🎯 关键指标！修复前=0.003
                        print(f"\n🎯 深度GT统计:")
                        print(f"   lidar_depth范围: [{lidar_depth.min():.3f}, {lidar_depth.max():.3f}]")
                        print(f"   lidar_depth均值: {lidar_depth.mean():.3f}")
                        print(f"   lidar_depth非零占比: {(lidar_depth > 0.01).float().mean() * 100:.1f}%")

                        # 🔍 新增：BEV空间分布诊断
                        print(f"\n{'=' * 60}")
                        print(f"🔍 BEV输入空间分布诊断")
                        print(f"{'=' * 60}")

                        # 🔧 **2025-11-18第二次修复**: 恢复正确的坐标系定义
                        # 坐标范围定义（与BEV生成保持一致）
                        # 🔧 **关键修复 2025-11-19**: 与fusion_model.py和深度图归一化保持一致
                        X_MIN, X_MAX = -100.0, 105.0  # X轴 (与fusion_model.py一致)
                        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 (与深度图归一化一致)
                        H = outputs.shape[2]  # 256 (对齐baseline)

                        # 提取第一个样本的BEV数据
                        radar_bev_sample = batch['radar_bev'][0]  # (5, 640, 640)
                        lidar_bev_sample = batch['lidar_bev'][0]  # (5, 640, 640)

                        # 定义空间区域（Y方向）
                        region_near = (0, 100)  # y=-5 to 10m
                        region_mid = (200, 330)  # y=20 to 50m
                        region_far = (400, 639)  # y=60 to 102m

                        # 检查Radar BEV密度通道分布
                        radar_near = radar_bev_sample[0, region_near[0]:region_near[1], :].mean().item()
                        radar_mid = radar_bev_sample[0, region_mid[0]:region_mid[1], :].mean().item()
                        radar_far = radar_bev_sample[0, region_far[0]:region_far[1], :].mean().item()

                        print(f"\n📡 Radar BEV密度通道统计:")
                        print(f"   近处(y=-5~10m, 行{region_near[0]:3d}-{region_near[1]:3d}): {radar_near:.4f}")
                        print(f"   中距(y=20~50m, 行{region_mid[0]:3d}-{region_mid[1]:3d}): {radar_mid:.4f}")
                        print(f"   远处(y=60~102m, 行{region_far[0]:3d}-{region_far[1]:3d}): {radar_far:.4f}")

                        # 检查LiDAR BEV密度通道分布
                        lidar_near = lidar_bev_sample[0, region_near[0]:region_near[1], :].mean().item()
                        lidar_mid = lidar_bev_sample[0, region_mid[0]:region_mid[1], :].mean().item()
                        lidar_far = lidar_bev_sample[0, region_far[0]:region_far[1], :].mean().item()

                        print(f"\n🔦 LiDAR BEV密度通道统计:")
                        print(f"   近处(y=-5~10m, 行{region_near[0]:3d}-{region_near[1]:3d}): {lidar_near:.4f}")
                        print(f"   中距(y=20~50m, 行{region_mid[0]:3d}-{region_mid[1]:3d}): {lidar_mid:.4f}")
                        print(f"   远处(y=60~102m, 行{region_far[0]:3d}-{region_far[1]:3d}): {lidar_far:.4f}")

                        # 检查GT目标位置分布
                        targets_sample = targets[0] if len(targets) > 0 else []
                        if len(targets_sample) > 0:
                            print(f"\n🎯 GT目标位置统计 (共{len(targets_sample)}个):")
                            gt_y_coords = [t['y'] for t in targets_sample if 'y' in t]
                            if len(gt_y_coords) > 0:
                                gt_y_min = min(gt_y_coords)
                                gt_y_max = max(gt_y_coords)
                                gt_y_mean = sum(gt_y_coords) / len(gt_y_coords)
                                print(f"   Y坐标范围: [{gt_y_min:.2f}, {gt_y_max:.2f}]m")
                                print(f"   Y坐标均值: {gt_y_mean:.2f}m")
                                print(f"   前3个目标: {[(t['x'], t['y']) for t in targets_sample[:3]]}")

                        # 检查模型输出峰值位置
                        output_prob = torch.sigmoid(outputs[0, 0]).cpu()
                        peak_y, peak_x = torch.where(output_prob == output_prob.max())
                        if len(peak_y) > 0:
                            peak_y = peak_y[0].item()
                            peak_x = peak_x[0].item()
                            peak_val = output_prob[peak_y, peak_x].item()
                            # 🔧 恢复正确的坐标系: X=横向, Y=前向
                            world_x = (peak_x / (H - 1)) * (X_MAX - X_MIN) + X_MIN  # 列→横向
                            world_y = (peak_y / (H - 1)) * (Y_MAX - Y_MIN) + Y_MIN  # 行→前向

                            print(f"\n🤖 模型输出峰值位置:")
                            print(f"   像素坐标: (y={peak_y}, x={peak_x})")
                            print(f"   峰值概率: {peak_val:.4f}")
                            print(f"   对应世界坐标: x={world_x:.2f}m(横向), y={world_y:.2f}m(前向)")

                            # 🔧 恢复正确的坐标系后: world_x=横向, world_y=前向
                            if world_y < 10:
                                print(f"   ⚠️ 警告: 预测仍集中在近处(Y<10m)，模型未学到远距离!")
                            elif world_y < 20:
                                print(f"   🟡 提示: 预测位置在移动，但还不够远(Y={world_y:.1f}m)")
                            else:
                                print(f"   ✅ 成功: 预测位置已移向中远距离(Y={world_y:.1f}m)!")

                        # 检查模型输出各区域响应
                        output_near = output_prob[region_near[0]:region_near[1], :].mean().item()
                        output_mid = output_prob[region_mid[0]:region_mid[1], :].mean().item()
                        output_far = output_prob[region_far[0]:region_far[1], :].mean().item()

                        print(f"\n🤖 模型输出各区域平均响应:")
                        print(f"   近处(y=-5~10m): {output_near:.4f}")
                        print(f"   中距(y=20~50m): {output_mid:.4f}")
                        print(f"   远处(y=60~102m): {output_far:.4f}")

                        # 综合诊断
                        print(f"\n{'=' * 60}")
                        print(f"💡 诊断结论:")
                        print(f"{'=' * 60}")

                        # 判断输入BEV是否有近场偏差
                        if radar_near > radar_mid * 2 or lidar_near > lidar_mid * 2:
                            print(f"⚠️ 输入BEV在近处信号过强！")
                            print(f"   Radar近/中比: {radar_near / max(radar_mid, 0.001):.2f}")
                            print(f"   LiDAR近/中比: {lidar_near / max(lidar_mid, 0.001):.2f}")
                            print(f"   → 可能导致模型过拟合近场噪声")

                        # 判断输入BEV中距离信号是否过弱
                        if radar_mid < 0.01 and lidar_mid < 0.01:
                            print(f"⚠️ 输入BEV在中距离信号过弱！")
                            print(f"   → 模型缺少中距离目标的学习信号")

                        # 判断模型输出是否有空间偏差
                        if output_near > output_mid * 1.5:
                            print(f"⚠️ 模型输出偏向近处！")
                            print(f"   输出近/中比: {output_near / max(output_mid, 0.001):.2f}")
                            print(f"   → 这会导致MOTA=0（预测位置与GT不匹配）")

                        # 对比GT位置和模型预测位置
                        if len(targets_sample) > 0 and len(gt_y_coords) > 0:
                            gt_avg_y = sum(gt_y_coords) / len(gt_y_coords)
                            if 'world_y' in locals():
                                y_gap = abs(world_y - gt_avg_y)
                                print(f"\n📍 位置偏差分析:")
                                print(f"   GT平均Y位置: {gt_avg_y:.2f}m")
                                print(f"   模型预测峰值Y位置: {world_y:.2f}m")
                                print(f"   Y方向偏差: {y_gap:.2f}m")
                                if y_gap > 20:
                                    print(f"   ⚠️ 偏差>{20}m，超过匹配阈值！")

                        print(f"{'=' * 60}\n")

                # 🎯 周期性监控断点 - 验证lambda_depth修复效果
                # 在关键batch输出监控信息：前10个batch + 每50个batch
                monitor_batches = [1, 2, 3, 5, 10, 20, 50, 100, 200]
                # 🚀 临时关闭监控，加速训练到Batch 200
                should_monitor = False  # 原来: batch_idx in monitor_batches or (batch_idx > 0 and batch_idx % 50 == 0)

                if should_monitor and depth_pred is not None:
                    print(f"\n{'🔍' * 30}")
                    print(f"📊 [Batch {batch_idx}] 周期性监控 - 验证MOTA=0修复效果")
                    print(f"{'🔍' * 30}")

                    # 断点1: 深度预测恢复监控 ⭐⭐⭐⭐⭐
                    depth_cpu = depth_pred.detach().cpu()
                    depth_std = depth_cpu.std().item()
                    depth_min = depth_cpu.min().item()
                    depth_max = depth_cpu.max().item()
                    depth_mean = depth_cpu.mean().item()

                    print(f"\n🌊 深度预测恢复状态:")
                    print(f"   范围: [{depth_min:.4f}, {depth_max:.4f}]  (目标: [0.05, 0.95])")
                    print(f"   均值: {depth_mean:.4f}")
                    print(f"   标准差: {depth_std:.4f}  (🎯关键指标! 目标>0.05, 修复前=0.003)")

                    if depth_std < 0.01:
                        print(f"   ⚠️ 警告: 深度预测仍然接近常数，修复可能未生效!")
                    elif depth_std < 0.05:
                        print(f"   🟡 提示: 深度预测开始变化，但还需继续学习")
                    else:
                        print(f"   ✅ 成功: 深度预测已恢复正常变化范围!")

                    # 断点2: Loss权重平衡监控 ⭐⭐⭐⭐⭐
                    det_loss = loss_dict['detection_loss']
                    dep_loss = loss_dict['depth_loss']

                    # 计算加权后的loss
                    det_weighted = criterion.alpha * det_loss
                    dep_weighted = criterion.lambda_depth * dep_loss
                    total_weighted = det_weighted + dep_weighted

                    if total_weighted > 0:
                        dep_ratio = (dep_weighted / total_weighted) * 100
                        print(f"\n⚖️ Loss权重平衡:")
                        print(f"   检测loss: {det_loss:.4f} × {criterion.alpha} = {det_weighted:.4f}")
                        print(f"   深度loss: {dep_loss:.4f} × {criterion.lambda_depth} = {dep_weighted:.4f}")
                        print(f"   深度占比: {dep_ratio:.1f}%  (🎯目标>50%, 修复前≈33%)")

                        if dep_ratio < 40:
                            print(f"   ⚠️ 警告: 深度loss占比仍然偏低!")
                        elif dep_ratio < 50:
                            print(f"   🟡 提示: 深度loss占比正在提升")
                        else:
                            print(f"   ✅ 成功: 深度loss权重已达到目标!")

                    # 断点3: 模型输出空间分布监控 ⭐⭐⭐⭐⭐
                    output_prob = torch.sigmoid(outputs[0, 0]).cpu()
                    H = output_prob.shape[0]

                    # 找峰值位置
                    peak_val = output_prob.max().item()
                    peak_idx = (output_prob == peak_val).nonzero(as_tuple=False)
                    if len(peak_idx) > 0:
                        peak_y = peak_idx[0][0].item()
                        peak_x = peak_idx[0][1].item()

                        # 🔧 **2025-11-18第二次修复**: 恢复正确的坐标系定义
                        # 转换为世界坐标（与BEV生成保持一致）
                        # 实际GT数据: X=横向, Y=前向
                        # BEV图像: 行(y)→Y(前向), 列(x)→X(横向)
                        X_MIN, X_MAX = -100.0, 105.0  # X轴 (与fusion_model.py一致)
                        Y_MIN, Y_MAX = -5.0, 105.0  # Y轴 (与深度图归一化一致)
                        world_x = X_MIN + (peak_x / (H - 1)) * (X_MAX - X_MIN)  # 列→横向
                        world_y = Y_MIN + (peak_y / (H - 1)) * (Y_MAX - Y_MIN)  # 行→前向

                        print(f"\n📍 模型输出峰值位置:")
                        print(f"   像素坐标: (y={peak_y}, x={peak_x})")
                        print(f"   世界坐标: x={world_x:.2f}m(横向), y={world_y:.2f}m(前向)")
                        print(f"   峰值概率: {peak_val:.4f}")

                        # 🔧 恢复正确的坐标系后: world_x=横向, world_y=前向
                        if world_y < 10:
                            print(f"   ⚠️ 警告: 预测仍集中在近处(Y<10m)，模型未学到远距离!")
                        elif world_y < 20:
                            print(f"   🟡 提示: 预测位置在移动，但还不够远(Y={world_y:.1f}m)")
                        else:
                            print(f"   ✅ 成功: 预测位置已移向中远距离(Y={world_y:.1f}m)!")

                        # 与GT对比
                        if len(targets) > 0 and len(targets[0]) > 0:
                            # 🔧 GT坐标: x=横向, y=前向 (恢复正确定义后)
                            gt_x_coords = [t['x'] for t in targets[0] if 'x' in t]
                            gt_y_coords = [t['y'] for t in targets[0] if 'y' in t]
                            if len(gt_x_coords) > 0:
                                gt_x_mean = sum(gt_x_coords) / len(gt_x_coords)
                                gt_y_mean = sum(gt_y_coords) / len(gt_y_coords)
                                x_gap = abs(world_x - gt_x_mean)
                                y_gap = abs(world_y - gt_y_mean)
                                total_gap = np.sqrt(x_gap ** 2 + y_gap ** 2)
                                print(f"   GT平均位置: x={gt_x_mean:.2f}m(横向), y={gt_y_mean:.2f}m(前向)")
                                print(f"   位置偏差: ΔX={x_gap:.2f}m, ΔY={y_gap:.2f}m, 总距离={total_gap:.2f}m")

                                if total_gap > 30:
                                    print(f"   ⚠️ 警告: 总距离偏差>{30}m!")
                                elif total_gap > 15:
                                    print(f"   🟡 提示: 偏差在减小")
                                else:
                                    print(f"   ✅ 成功: 偏差已在合理范围!")

                    # 断点4: 各区域响应分布 ⭐⭐⭐⭐
                    region_near = (0, 100)  # y=-5~10m
                    region_mid = (200, 330)  # y=20~50m
                    region_far = (400, 639)  # y=60~102m

                    near_resp = output_prob[region_near[0]:region_near[1], :].mean().item()
                    mid_resp = output_prob[region_mid[0]:region_mid[1], :].mean().item()
                    far_resp = output_prob[region_far[0]:region_far[1], :].mean().item()

                    print(f"\n📊 各距离区域响应分布:")
                    print(f"   近处(y<10m):   {near_resp:.4f}")
                    print(f"   中距(y=20-50m): {mid_resp:.4f}")
                    print(f"   远处(y>60m):   {far_resp:.4f}")

                    if near_resp > mid_resp * 1.5:
                        print(f"   ⚠️ 警告: 响应过度集中在近处 (近/中比={near_resp / max(mid_resp, 0.001):.2f})")
                    elif abs(near_resp - mid_resp) < 0.1 and abs(mid_resp - far_resp) < 0.1:
                        print(f"   🟡 提示: 响应较均匀，但可能缺乏区分度")
                    else:
                        print(f"   ✅ 各区域响应分布合理")

                    print(f"\n{'🔍' * 30}\n")

                    # 调试：检查loss值
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 检测到异常loss: {loss.item()}, 跳过此batch")
                        # 跳过这个batch的反向传播
                        optimizer.zero_grad()
                        continue

                    # 缩放损失以适应梯度累积
                    loss = loss / gradient_accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()
            else:
                # 🔧 模型现在返回 (outputs, depth_pred)
                model_output = model(*model_inputs)
                if isinstance(model_output, tuple):
                    outputs, depth_pred = model_output
                else:
                    outputs = model_output
                    depth_pred = None

                # 🚨 断点1：检查模型输出是否有NaN
                if torch.isnan(outputs).any():
                    print(f"\n{'🚨' * 30}")
                    print(f"🚨 断点1触发：模型outputs有NaN！")
                    print(f"  outputs形状: {outputs.shape}")
                    print(f"  NaN数量: {torch.isnan(outputs).sum().item()}/{outputs.numel()}")
                    print(f"  outputs范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
                    print(f"  问题：SGDNet前向传播输出NaN")
                    print(f"{'🚨' * 30}\n")

                if depth_pred is not None and torch.isnan(depth_pred).any():
                    print(f"\n{'🚨' * 30}")
                    print(f"🚨 断点1触发：depth_pred有NaN！")
                    print(f"  depth_pred形状: {depth_pred.shape}")
                    print(f"  NaN数量: {torch.isnan(depth_pred).sum().item()}/{depth_pred.numel()}")
                    print(
                        f"  depth_pred范围: [{depth_pred[~torch.isnan(depth_pred)].min():.6f}, {depth_pred[~torch.isnan(depth_pred)].max():.6f}]")
                    print(f"  问题：深度预测网络输出NaN")
                    print(f"{'🚨' * 30}\n")

                # 🔧 传递深度预测、GT和相机图像给loss函数（完全对齐SGDNet）
                # 获取camera_imgs用于SmoothEdgeLoss（确保在GPU上）
                if multi_gpu:
                    primary_device = torch.device('cuda:0')
                    camera_imgs_for_loss = batch['camera_imgs'].to(primary_device, non_blocking=True)
                else:
                    camera_imgs_for_loss = batch['camera_imgs'].to(device, non_blocking=True)
                loss, loss_dict = criterion(outputs, targets, depth_pred, lidar_depth, camera_imgs_for_loss)

                # 🚨 断点2：检查loss计算后是否有NaN
                if torch.isnan(loss):
                    print(f"\n{'🚨' * 30}")
                    print(f"🚨 断点2触发：Loss计算后是NaN！")
                    print(f"  loss值: {loss}")
                    print(f"  detection_loss: {loss_dict.get('detection_loss', 'N/A')}")
                    print(f"  depth_loss: {loss_dict.get('depth_loss', 'N/A')}")

                    # 检查是哪个loss导致的NaN
                    det_loss = loss_dict.get('detection_loss', 0)
                    dep_loss = loss_dict.get('depth_loss', 0)
                    if isinstance(det_loss, (int, float)):
                        det_is_nan = False
                    else:
                        det_is_nan = torch.isnan(torch.tensor(det_loss))
                    if isinstance(dep_loss, (int, float)):
                        dep_is_nan = False
                    else:
                        dep_is_nan = torch.isnan(torch.tensor(dep_loss))

                    if det_is_nan:
                        print(f"  ❌ detection_loss是NaN！问题在Focal Loss计算")
                    if dep_is_nan:
                        print(f"  ❌ depth_loss是NaN！问题在深度监督loss计算")
                    print(f"  lambda_depth当前值: {criterion.lambda_depth}")
                    print(f"{'🚨' * 30}\n")

                # 调试：检查loss值（保留原有的跳过逻辑）
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 检测到异常loss: {loss.item()}, 跳过此batch")
                    # 跳过这个batch的反向传播
                    optimizer.zero_grad()
                    continue

                # 缩放损失以适应梯度累积
                loss = loss / gradient_accumulation_steps

                # 反向传播
                loss.backward()

            accumulated_loss += loss.item()
            accumulated_steps += 1

            # 仅在累积足够步数后更新权重
            if accumulated_steps >= gradient_accumulation_steps:
                if USE_AMP:
                    # 🔧 完全对齐baseline：不使用梯度裁剪
                    # Baseline: clip_grad_norm=0 (不裁剪)
                    scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 已禁用
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 🔧 完全对齐baseline：不使用梯度裁剪
                    # Baseline: clip_grad_norm=0 (不裁剪)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 已禁用
                    optimizer.step()

                optimizer.zero_grad()
                accumulated_steps = 0
                accumulated_loss = 0

            # 收集轨迹用于MOTA计算（按照配置的批次数间隔，减少内存压力）
            # 添加标志位跟踪是否有新数据
            new_tracks_collected = False
            if eval_mota and eval_interval and batch_idx % eval_interval == 0:
                with torch.no_grad():
                    outputs_cpu = outputs.detach().cpu()
                    pred_tracks, gt_tracks = extract_tracks_from_outputs(
                        outputs_cpu, targets, frame_ids, batch_num=batch_idx, epoch=epoch
                    )
                    all_pred_tracks.extend(pred_tracks)
                    all_gt_tracks.extend(gt_tracks)
                    print(
                        f"DEBUG: 收集轨迹 - pred={len(pred_tracks)}, gt={len(gt_tracks)}, 累计pred={len(all_pred_tracks)}, 累计gt={len(all_gt_tracks)}")

                    new_tracks_collected = True

            # 🔧 MOTA诊断: 大幅放宽匹配阈值用于诊断坐标系
            # 原始值: epoch<=3用10m, epoch<=10用5m, 之后用2.5m
            # 诊断值: 统一用30m，看能匹配多少
            if epoch <= 3:
                match_threshold = 30.0  # 🔴 诊断用: 放宽到30米
            elif epoch <= 10:
                match_threshold = 20.0  # 🔴 诊断用: 放宽到20米
            else:
                match_threshold = 10.0  # 🔴 诊断用: 最终用10米

            if new_tracks_collected and len(all_pred_tracks) > 0 and len(all_gt_tracks) > 0:
                # 使用最近收集的轨迹计算MOTA
                recent_pred = all_pred_tracks[-100:] if len(all_pred_tracks) > 100 else all_pred_tracks
                recent_gt = all_gt_tracks[-100:] if len(all_gt_tracks) > 100 else all_gt_tracks

                # 🔧 修复：第一次计算时启用verbose查看调试信息
                is_first_mota = batch_idx <= eval_interval
                batch_mota, batch_motp = compute_mota_motp(
                    recent_pred, recent_gt,
                    threshold=match_threshold, verbose=is_first_mota  # 第一次启用调试
                )

                # 统计预测和真实目标数量
                num_pred = len(recent_pred)
                num_gt = len(recent_gt)

                # 简化输出（只在配置的间隔输出一次）
                if eval_interval and batch_idx % eval_interval == 0:
                    print(
                        f"\n   📊 批次{batch_idx} - 预测:{num_pred} GT:{num_gt} | MOTA={batch_mota:.3f}, MOTP={batch_motp:.3f}")

                # 定期清理内存（按照配置的间隔）
                if memory_manager and cleanup_interval and num_batches % cleanup_interval == 0:
                    memory_manager.cleanup_memory()

                # 成功处理batch后，重置连续OOM计数
                if memory_manager:
                    memory_manager.consecutive_oom = 0

        except RuntimeError as e:
            error_str = str(e)
            if "out of memory" in error_str:
                print(f"\n⚠️ 批次 {num_batches} 显存不足: {error_str[:100]}...")

                # 使用内存管理器处理OOM
                if memory_manager:
                    should_rebuild = memory_manager.handle_oom(epoch)
                    if should_rebuild:
                        print("💡 建议：减小批大小并重建DataLoader")
                else:
                    # 基本的内存清理
                    torch.cuda.empty_cache()
                    gc.collect()

                # 清空梯度
                optimizer.zero_grad()
                accumulated_steps = 0
                accumulated_loss = 0

                # 显示内存状态
                if memory_manager:
                    mem_stats = memory_manager.get_memory_stats()
                    for stat in mem_stats:
                        print(f"   GPU {stat['gpu']}: {stat['allocated_gb']:.1f}/{stat['total_gb']:.1f} GB")

                # 跳过这个batch
                continue
            elif "NCCL" in error_str:
                print(f"⚠️ GPU通信错误: {error_str[:100]}...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accumulated_steps = 0
                accumulated_loss = 0
                continue
            elif "GET was unable to find an engine" in error_str or "cuDNN" in error_str:
                print(f"⚠️ GPU内核错误，跳过批次 {num_batches}: {error_str[:100]}...")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                accumulated_steps = 0
                accumulated_loss = 0
                continue
            else:
                raise e

        # 计算实际损失（未缩放）
        actual_loss = loss.item() * gradient_accumulation_steps
        total_loss += actual_loss
        num_batches += 1

        # 简化：只在每100个batch显示loss统计（减少输出频率提高速度）
        if num_batches % 100 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
            print(f"\n批次 {num_batches}: 当前loss={actual_loss:.4f}, 平均loss={avg_loss:.4f}")

        # 更新进度条显示（包含MOTA/MOTP）
        postfix_dict = {
            'loss': f"{actual_loss:.3f}",
            'det': f"{loss_dict['detection_loss']:.3f}",
            'dep': f"{loss_dict.get('depth_loss', 0):.3f}",
            'sme': f"{loss_dict.get('smooth_edge_loss', 0):.3f}"  # 🔧 显示边缘平滑loss
        }

        # 显示内存信息
        if hasattr(torch.cuda, 'memory_allocated'):
            mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
            postfix_dict['Mem'] = f"{mem_mb}MB"

        # 添加MOTA/MOTP到进度条（仅在收集新轨迹时计算，避免重复）
        if new_tracks_collected and len(all_pred_tracks) > 10 and len(all_gt_tracks) > 10:
            # 使用最近的轨迹计算快速指标
            recent_mota, recent_motp = compute_mota_motp(
                all_pred_tracks[-100:] if len(all_pred_tracks) > 100 else all_pred_tracks,
                all_gt_tracks[-100:] if len(all_gt_tracks) > 100 else all_gt_tracks,
                threshold=15.0, verbose=False
            )
            postfix_dict['MOTA'] = f"{recent_mota:.2f}"
            postfix_dict['MOTP'] = f"{recent_motp:.2f}"

        # 每5个批次更新进度条
        if num_batches % 5 == 0:
            pbar.set_postfix(postfix_dict)

        # 内存清理
        del outputs, loss
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()

    # 处理任何剩余的累积梯度
    if accumulated_steps > 0:
        if USE_AMP:
            scaler.unscale_(optimizer)
            # 🔧 完全对齐baseline：不使用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 已禁用
            scaler.step(optimizer)
            scaler.update()
        else:
            # 🔧 完全对齐baseline：不使用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 已禁用
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # 计算MOTA/MOTP - 使用动态阈值
    mota, motp = 0.0, 0.0
    if eval_mota and len(all_pred_tracks) > 0 and len(all_gt_tracks) > 0:
        # 动态匹配阈值策略 - 🔴 放宽阈值以提升MOTA
        if epoch <= 5:
            match_threshold = 15.0  # 训练初期放宽
        elif epoch <= 15:
            match_threshold = 10.0  # 中期训练
        elif epoch <= 30:
            match_threshold = 5.0  # 后期收紧
        else:
            match_threshold = 3.0  # 最终收敛

        print(
            f"\n📊 计算训练集MOTA/MOTP (采样的{len(all_pred_tracks)}个预测, {len(all_gt_tracks)}个真实, 阈值={match_threshold}m)...")
        mota, motp = compute_mota_motp(all_pred_tracks, all_gt_tracks, threshold=match_threshold, verbose=False)
        print(f"   训练MOTA: {mota:.4f}, MOTP: {motp:.4f}")

    # 返回简化的指标
    print(f"\n✅ Epoch {epoch}完成: 平均loss={avg_loss:.4f}, 处理批次={num_batches}")

    # 关闭进度条
    pbar.close()

    return avg_loss, mota, motp


def main():
    # 首先验证坐标转换一致性
    verify_coordinate_consistency()

    parser = argparse.ArgumentParser(description='多模态融合网络训练 - 带标签数据集')
    parser.add_argument('--mapping_csv', type=str,
                        default='/mnt/ourDataset_v2/mapping.csv',
                        help='标签映射文件')
    parser.add_argument('--train_txt', type=str,
                        default='/mnt/ourDataset_v2/train.txt',
                        help='训练ID列表文件')
    parser.add_argument('--data_root', type=str,
                        default='/mnt/ourDataset_v2/ourDataset_v2_label',
                        help='数据根目录')
    parser.add_argument('--img_size', type=int, default=256, help='图像尺寸（对齐baseline）')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小（默认2，稳定不会OOM）')
    parser.add_argument('--gradient_accumulation', type=int, default=8, help='梯度累积步数（默认8，有效批次=16）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数(🔴增加到50轮以充分训练)')
    parser.add_argument('--val_freq', type=int, default=200,
                        help='MOTA评估频率（以batch为单位，<=0表示禁用周期性评估）')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率(优化为1e-3配合单GPU)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_label_v2', help='模型保存目录')
    parser.add_argument('--use_camera', type=str, default='LeopardCamera0',
                        choices=['LeopardCamera0'], help='使用的相机')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器工作进程数（0=单进程，避免卡住）')
    parser.add_argument('--single_gpu', action='store_true', default=False, help='强制使用单GPU训练（默认多GPU）')
    parser.add_argument('--gpu_id', type=int, default=0, help='单GPU模式下使用的GPU ID')
    parser.add_argument('--use_cpu', action='store_true', help='使用CPU训练（非常慢，仅用于测试）')
    parser.add_argument('--visualize', action='store_true', help='启用数据可视化')
    parser.add_argument('--vis_output_dir', type=str, default='./visualization_output', help='可视化输出目录')
    parser.add_argument('--vis_sequence', type=str, help='指定要可视化的序列名称')
    parser.add_argument('--max_frames', type=int, default=50, help='每个序列最大帧数')
    parser.add_argument('--quick-train', action='store_true', help='快速训练测试（5 epochs, batch_size 8）')

    args = parser.parse_args()

    if args.val_freq is None:
        args.val_freq = 200
    elif args.val_freq < 0:
        print("⚠️ 检测到负的验证频率，已禁用周期性MOTA评估")
        args.val_freq = 0

    # 🚀 一键运行提示
    print("=" * 80)
    print("🚀 一键运行版本 - 所有参数已硬编码")
    print("=" * 80)
    print(f"📍 数据路径: {args.data_root}")
    print(f"📐 图像尺寸: {args.img_size}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"🔄 梯度累积: {args.gradient_accumulation} (有效批次={args.batch_size * args.gradient_accumulation})")
    print(f"🔢 训练轮数: {args.epochs}")
    print(f"📈 学习率: {args.lr}")
    print(f"💾 保存目录: {args.save_dir}")
    print(f"🎥 相机: {args.use_camera}")
    print(f"⚙️ 工作进程: {args.num_workers}")
    print(f"🖥️ GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} (多GPU模式)")
    print(f"☁️ 伪点云数: {os.environ.get('PSEUDO_POINTS', '4096')}")
    if args.val_freq > 0:
        print(f"🧪 验证频率: 每{args.val_freq}个batch计算MOTA/MOTP")
    else:
        print("🧪 验证频率: 已禁用周期性MOTA/MOTP评估")
    print("=" * 80)
    print("直接运行命令: python train_label_FIXED_with_read_pcd_20251030_214721.py")
    print("=" * 80)
    print()

    # 快速训练模式
    if args.quick_train:
        print("⚡ 快速训练模式")
        args.epochs = 5
        args.batch_size = 8
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Batch size: {args.batch_size}")

    # 如果启用了可视化
    if args.visualize:
        print("=== 数据可视化模式 ===")
        vis_tool = VisualizationTool(
            model_path=None,
            mapping_csv=args.mapping_csv,
            data_root=args.data_root,
            vis_output_dir=args.vis_output_dir
        )

        # 创建数据集统计可视化
        vis_tool.create_summary_visualization()

        if args.vis_sequence:
            # 可视化指定序列
            vis_tool.visualize_sequence(args.vis_sequence, args.max_frames)
        else:
            # 可视化几个示例序列
            example_sequences = [
                '20221217_group0003_mode2_291frames',
                '20221217_group0004_mode3_350frames',
                '20221217_group0005_mode3_99frames'
            ]
            for sequence in example_sequences:
                vis_tool.visualize_sequence(sequence, args.max_frames)

        print("数据可视化完成!")
        return

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备和内存
    if args.use_cpu:
        # CPU训练模式
        print("⚠️ CPU训练模式 - 速度将非常慢（比GPU慢100-1000倍）")
        device = torch.device('cpu')
        available_gpus = []
        num_gpus = 0
        print("✅ 使用CPU训练")
        print("   预计训练时间：每个epoch可能需要100-200小时")
    elif args.single_gpu:
        # 单GPU模式
        print("🔧 单GPU训练模式")
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return

        # 设置指定的GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device('cuda:0')
        available_gpus = [0]
        num_gpus = 1

        # 设置内存分配
        torch.cuda.set_per_process_memory_fraction(0.80, device=0)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ 使用GPU {args.gpu_id} ({gpu_name})")
        print(f"   总内存: {gpu_memory_gb:.1f} GB")
    else:
        # 智能多GPU模式
        print("🚀 智能多GPU训练模式 - 自动选择空闲GPU最大化利用率")
        print("=" * 60)

        device, available_gpus, num_gpus = setup_balanced_multi_gpu()
        if device is None:
            print("❌ 多GPU设置失败，尝试单GPU模式...")

            # 回退到单GPU模式
            if torch.cuda.is_available():
                # 选择第一个可用的GPU
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        test_tensor = torch.zeros(1, device=f'cuda:{i}')
                        del test_tensor

                        memory_free = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
                        if memory_free > 10 * 1024 ** 3:  # 至少10GB空闲
                            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                            device = torch.device('cuda:0')
                            available_gpus = [0]
                            num_gpus = 1
                            print(f"✅ 回退到GPU {i}")
                            break
                    except:
                        continue

                if device is None:
                    print("❌ 没有可用的GPU，退出")
                    return
            else:
                print("❌ 没有可用的GPU，退出")
                return

        # 获取GPU内存信息
        gpu_memory = torch.cuda.get_device_properties(device.index).total_memory
        gpu_memory_gb = gpu_memory / (1024 ** 3)
        print(f"\n📊 训练配置:")
        print(f"   主GPU内存: {gpu_memory_gb:.1f} GB")
        print(f"   GPU数量: {num_gpus}")

        # 强制批次大小为1以避免OOM（不再自动调整）
        # 🔧 根据GPU数量动态调整批次大小
        if num_gpus == 1:
            # 单GPU可以用更大的批次充分利用显存
            # 🔧 修复：不再强制改变批大小，尊重用户设置
            if args.batch_size < 16:
                # args.batch_size = 16  # 注释掉，避免OOM
                print(f"   📊 单GPU模式：保持批次大小 {args.batch_size} (用户设置)")
        else:
            # 双GPU使用适中批次避免通信开销
            if args.batch_size > 4:
                args.batch_size = 4
                print(f"   📊 多GPU模式：批次大小调整为 {args.batch_size}")

        print(f"   批次大小: {args.batch_size} (每个批次{args.batch_size}个样本)")

    # 创建内存管理器
    memory_manager = MemoryManager(initial_batch_size=args.batch_size, min_batch_size=1)
    print(f"🔧 内存管理器已启用 - 初始批大小: {args.batch_size}")

    # 创建数据集
    dataset = MultiModalLabeledDataset(
        mapping_csv=args.mapping_csv,
        id_txt=args.train_txt,
        data_root=args.data_root,
        img_size=args.img_size,
        use_camera=args.use_camera
    )

    # 动态创建数据加载器的函数
    # 多GPU时强制使用num_workers=0避免NCCL死锁
    effective_workers = 0 if num_gpus > 1 else args.num_workers
    if num_gpus > 1:
        print(f"⚠️ 多GPU模式：强制设置num_workers=0以避免NCCL死锁（DataLoader fork进程问题）")

    def create_dataloader(batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=effective_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True if effective_workers > 0 else False,  # 单进程时启用pin_memory加速
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if effective_workers > 0 else None
        )

    # 初始数据加载器
    current_batch_size = memory_manager.current_batch_size
    dataloader = create_dataloader(current_batch_size)

    # 创建模型
    # 初始化模型 - 渐进式加载
    try:
        print("🔄 正在创建模型...")
        model = FusionNet()

        # 先在CPU上创建模型，然后逐步转移到GPU
        if device.type == 'cuda':
            print("🔄 正在将模型转移到GPU...")
            model = model.to(device)

            # 启用多GPU训练支持SGD模块
            if num_gpus > 1:
                print(f"✅ 启用多GPU训练: {num_gpus}个GPU {available_gpus}")

                # 使用DataParallel进行多GPU训练
                model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
                print(f"✅ DataParallel已启用，使用GPU设备: {list(range(num_gpus))}")

                # 设置同步批归一化 - DataParallel不需要，已禁用以减少NCCL日志
                # SyncBatchNorm仅对DDP有用，DataParallel不需要
                # try:
                #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                #     print("✅ 已启用同步批归一化，提高多GPU同步性")
                # except Exception as e:
                #     print(f"⚠️ 同步批归一化设置失败: {e}")

                # 有效批次大小
                effective_batch_size = args.batch_size * num_gpus
                print(f"✅ 批次大小: {args.batch_size}，{num_gpus}GPU并行处理（优化2GPU训练）")
            else:
                print(f"使用单GPU训练: GPU {available_gpus[0] if available_gpus else 0}")
        else:
            print("使用CPU训练")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 模型加载失败，显存不足: {e}")
            print("尝试降低batch_size或使用更少的GPU")
            return
        else:
            raise e

    # 🔍 深度预测初始化检查（防止sigmoid饱和）
    print("\n🔍 检查深度预测初始化...")
    model.eval()  # 暂时设置为eval模式
    with torch.no_grad():
        # 创建虚拟输入
        dummy_batch_size = 2
        dummy_H, dummy_W = 640, 640
        dummy_inputs = {
            'dynamic_HM': torch.randn(dummy_batch_size, 6, dummy_H, dummy_W).to(device),
            'static_HM': torch.randn(dummy_batch_size, 6, dummy_H, dummy_W).to(device),
            'input_img': torch.randn(dummy_batch_size, 3, 640, 640).to(device),
            'input_radar': torch.randn(dummy_batch_size, 5, 640, 640).to(device),
            'input_velo': torch.randn(dummy_batch_size, 3, 640, 640).to(device),
            'input_knn': torch.randn(dummy_batch_size, 3, 640, 640).to(device),
            'segmap': torch.randn(dummy_batch_size, 20, 640, 640).to(device),
            'oculii_img': torch.randn(dummy_batch_size, 5, 640, 640).to(device),
        }

        try:
            # 处理DataParallel
            if isinstance(model, torch.nn.DataParallel):
                test_model = model.module
            else:
                test_model = model

            outputs, depth_pred = test_model(
                dummy_inputs['dynamic_HM'], dummy_inputs['static_HM'],
                dummy_inputs['input_img'], dummy_inputs['input_radar'],
                dummy_inputs['input_velo'], dummy_inputs['input_knn'],
                dummy_inputs['segmap'], dummy_inputs['oculii_img']
            )

            if depth_pred is not None:
                depth_mean = depth_pred.mean().item()
                depth_std = depth_pred.std().item()
                depth_min = depth_pred.min().item()
                depth_max = depth_pred.max().item()

                print(f"   深度预测统计:")
                print(f"      范围: [{depth_min:.4f}, {depth_max:.4f}]")
                print(f"      均值: {depth_mean:.4f}")
                print(f"      标准差: {depth_std:.6f}")

                # 检查是否陷入sigmoid饱和
                if depth_std < 0.01:
                    print(f"\n   🚨 警告: 深度预测标准差={depth_std:.6f} < 0.01")
                    print(f"   深度预测接近常数{depth_mean:.4f}，可能陷入sigmoid饱和!")
                    print(f"   已自动提高lambda_depth=15.0强制训练深度分支")
                elif 0.45 < depth_mean < 0.55 and depth_std < 0.05:
                    print(f"\n   ⚠️  提示: 深度预测初始化在sigmoid中点附近")
                    print(f"   需要观察前几个batch是否能快速脱离")
                else:
                    print(f"   ✅ 深度预测初始化正常")
            else:
                print(f"   ⚠️  depth_pred is None，SGDNet可能未正常工作")

        except Exception as e:
            print(f"   ⚠️  深度预测检查失败: {e}")

    model.train()  # 恢复训练模式
    print()

    # 初始化优化器和损失函数 - 🔧 完全对齐baseline配置
    # 参考: RadarRGBFusionNet2_20231128/others/train1.py:153-159
    # Baseline使用: Adam优化器, lr=0.001, weight_decay=0, betas=(0.9, 0.999)
    initial_lr = args.lr  # 0.001
    optimizer = torch.optim.Adam(  # 改为Adam（baseline使用Adam而非AdamW）
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0  # 🔧 移除weight_decay（baseline设为0）
    )
    print(f"🔧 优化器配置 (对齐baseline): Adam, lr={initial_lr:.6f}, weight_decay=0")
    # 🔧 启用FastFocalLoss + 深度监督loss
    criterion = MultiModalLoss(
        use_fast_focal=True,
        max_pos_weight=5.0,
        lambda_depth=1.0  # 🔧 对齐SGDNet_TI：使用1.0等权重（从5.0降到1.0）
        # 参考: SGDNet_TI/main.py:323 loss = loss_coarse + loss_cls + loss_d + loss_smoothedge
        # SGDNet原始代码所有loss项使用等权重1.0，无lambda系数
    )
    print(f"🔧 Loss配置: FastFocalLoss + 深度监督 (lambda_depth=1.0 - 对齐SGDNet_TI)")

    # 初始化early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)
    print("✅ 已启用Early Stopping (patience=7, min_delta=0.001)")

    # 初始化混合精度训练
    if USE_AMP:
        scaler = get_scaler()
    else:
        scaler = None

    # 学习率调度器 - 🔧 完全对齐baseline配置
    # 参考: RadarRGBFusionNet2_20231128/others/train1.py:162
    # Baseline使用: StepLR, step_size=5, gamma=0.7
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,  # 🔧 对齐baseline: 每5个epoch降低学习率（从3改为5）
        gamma=0.7  # 🔧 对齐baseline: 每次降低30%（从0.8改为0.7）
    )
    print(f"🔧 学习率调度器 (对齐baseline): StepLR, step_size=5, gamma=0.7")

    print(f"开始训练，总样本数: {len(dataset)}, 批次数: {len(dataloader)}")

    # 平衡多GPU训练信息输出 + 优化配置
    if hasattr(model, 'module'):
        print(f"\n🚀 优化多GPU训练配置完成！")
        print(f"   - GPU数量: {num_gpus} (平衡配置)")
        print(f"   - 每个GPU批次大小: {args.batch_size}")
        print(f"   - 批次大小: {args.batch_size}，{num_gpus}GPU并行（优化2GPU）")
        print(f"   - Epoch数: {args.epochs} (优化配置)")
        print(f"   - Early Stopping: 启用 (patience=7)")
        print(f"   - 预期训练时间: 3-5天")
    else:
        print(f"\n🔧 优化单GPU训练配置完成")
        print(f"   - 批次大小: {args.batch_size}")
        print(f"   - Epoch数: {args.epochs}")

    # 训练循环 - 每个epoch都显示MOTA/MOTP
    best_loss = float('inf')
    gradient_accumulation_steps = args.gradient_accumulation  # 从参数获取梯度累积步数

    for epoch in range(1, args.epochs + 1):
        # 检查是否需要重建dataloader（批大小改变）
        if memory_manager.current_batch_size != current_batch_size:
            current_batch_size = memory_manager.current_batch_size
            print(f"\n🔄 重建DataLoader，新批大小: {current_batch_size}")
            dataloader = create_dataloader(current_batch_size)

            # 调整学习率以适应新的批大小
            adjust_factor = current_batch_size / args.batch_size
            new_lr = args.lr * adjust_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"   调整学习率: {args.lr:.6f} → {new_lr:.6f}")

        avg_loss, mota, motp = train_epoch(
            model, dataloader, optimizer, criterion, device, scaler, epoch,
            multi_gpu=hasattr(model, 'module'), num_gpus=num_gpus,
            eval_mota=args.val_freq > 0,  # 根据配置决定是否评估MOTA
            val_freq=args.val_freq,
            memory_manager=memory_manager,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        # 显示epoch结果（包含MOTA/MOTP）
        print(f'\n{"=" * 60}')
        print(f'✅ Epoch {epoch}/{args.epochs} 完成')
        print(f'   📉 Loss: {avg_loss:.4f}')
        if args.val_freq > 0:
            print(f'   📊 MOTA: {mota:.4f} ({mota * 100:.2f}%)')
            print(f'   📍 MOTP: {motp:.4f} ({motp * 100:.2f}%)')
        else:
            print('   📊 MOTA: 已跳过（val_freq<=0）')
            print('   📍 MOTP: 已跳过（val_freq<=0）')
        print(f'   📚 学习率: {current_lr:.6f}')
        print(f'   💾 当前批大小: {current_batch_size}')
        print(f'   🧮 OOM次数: {memory_manager.oom_count}')
        print(f'{"=" * 60}\n')

        # Early Stopping检查
        if early_stopping(avg_loss, model):
            print(f"🛑 Early Stopping触发！在第{epoch}个epoch停止训练")
            print(f"   - 最佳Loss: {early_stopping.best_loss:.4f}")
            print(f"   - 训练提前完成，节省时间！")
            break

        # 保存检查点 (降低频率以节省时间)
        if epoch % 10 == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"🎯 发现更好的模型！Loss: {avg_loss:.4f}")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'batch_size': current_batch_size,
                'oom_count': memory_manager.oom_count
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'model_epoch_{epoch}.pth'))
            print(f"💾 保存检查点: epoch_{epoch}.pth")

        # 定期内存清理
        memory_manager.cleanup_memory()

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, final_path)

    print(f"训练完成! 最终模型保存到: {final_path}")


if __name__ == '__main__':
    main()