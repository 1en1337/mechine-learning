import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
import pickle
from collections import OrderedDict
import threading
from typing import Optional, Tuple, List
from .normalization import SpectralNormalizer, NormalizationMethod


class IsolatedSpectralDataset(Dataset):
    """
    改进的光谱数据集，确保训练集和验证集之间的缓存隔离
    """
    def __init__(self, data_path, cache_size=100, transform=None, 
                 normalize=True, normalization_method='mean_std',
                 dataset_type='train', cache_dir='.cache'):
        """
        Args:
            data_path: 数据文件夹路径
            cache_size: 缓存大小
            transform: 数据变换
            normalize: 是否归一化
            normalization_method: 归一化方法
            dataset_type: 'train' 或 'val'，用于缓存隔离
            cache_dir: 缓存目录
        """
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        self.transform = transform
        self.dataset_type = dataset_type
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        # 为不同数据集类型创建独立的缓存
        self.cache_dir = Path(cache_dir) / dataset_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 独立的索引缓存路径
        self.index_cache_path = self.cache_dir / f'{dataset_type}_index_cache.pkl'
        
        # 使用OrderedDict确保FIFO顺序
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        
        # 数据统计信息（用于归一化）
        self._stats = None
        self._stats_lock = threading.Lock()
        
        # 构建文件索引
        self._build_index()
        
        # 计算数据集统计信息
        if self.normalizer.method == NormalizationMethod.MEAN_STD:
            self._compute_stats()
    
    def _build_index(self):
        """构建文件索引，支持缓存和验证"""
        if self.index_cache_path.exists():
            print(f"Loading {self.dataset_type} index from cache: {self.index_cache_path}")
            try:
                with open(self.index_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # 验证缓存数据
                if self._validate_cache(cached_data):
                    self.file_list = cached_data['file_list']
                    self._stats = cached_data.get('stats', None)
                    print(f"Successfully loaded {len(self.file_list)} files from cache")
                    return
                    
            except Exception as e:
                print(f"Cache loading failed ({e}), rebuilding index...")
                
        self._build_fresh_index()
    
    def _validate_cache(self, cached_data):
        """验证缓存数据的完整性"""
        if not isinstance(cached_data, dict):
            return False
            
        if 'file_list' not in cached_data:
            return False
            
        file_list = cached_data['file_list']
        if not isinstance(file_list, list) or len(file_list) == 0:
            return False
            
        # 检查至少一些文件是否存在
        sample_size = min(10, len(file_list))
        sample_indices = np.random.choice(len(file_list), sample_size, replace=False)
        
        for idx in sample_indices:
            if not Path(file_list[idx]).exists():
                return False
                
        return True
    
    def _build_fresh_index(self):
        """重新构建文件索引"""
        print(f"Building {self.dataset_type} file index...")
        self.file_list = []
        
        # 递归搜索所有H5和NPY文件
        for pattern in ['*.h5', '*.npy']:
            for file_path in self.data_path.rglob(pattern):
                self.file_list.append(str(file_path))
        
        print(f"Found {len(self.file_list)} files")
        
        if len(self.file_list) == 0:
            raise ValueError(f"No data files found in {self.data_path}")
        
        # 排序以确保一致性
        self.file_list.sort()
        
        # 保存索引缓存
        self._save_index_cache()
    
    def _save_index_cache(self):
        """保存索引缓存"""
        try:
            cache_data = {
                'file_list': self.file_list,
                'stats': self._stats,
                'dataset_type': self.dataset_type,
                'data_path': str(self.data_path)
            }
            
            with open(self.index_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"{self.dataset_type} index cache saved")
        except Exception as e:
            print(f"Failed to save index cache: {e}")
    
    def _compute_stats(self):
        """计算数据集统计信息用于归一化"""
        stats_path = self.cache_dir / f'{self.dataset_type}_stats.pkl'
        
        if stats_path.exists():
            try:
                self.normalizer.load_stats(stats_path)
                print(f"Loaded {self.dataset_type} statistics from cache")
                return
            except Exception as e:
                print(f"Failed to load statistics: {e}")
        
        print(f"Computing {self.dataset_type} dataset statistics...")
        
        # 采样计算统计信息
        sample_size = min(1000, len(self.file_list))
        sample_indices = np.random.choice(len(self.file_list), sample_size, replace=False)
        
        samples = []
        for idx in sample_indices:
            try:
                lyso, hpge = self._load_file(idx)
                if lyso is not None and hpge is not None:
                    samples.append((lyso, hpge))
            except Exception:
                continue
        
        if samples:
            # 计算并保存统计信息
            self.normalizer.compute_stats(samples)
            self.normalizer.save_stats(stats_path)
            print(f"{self.dataset_type} statistics computed and saved")
        else:
            print(f"Warning: Could not compute statistics, using defaults")
            # 设置默认统计信息
            self.normalizer.stats = {
                'lyso_mean': 0.0,
                'lyso_std': 1.0,
                'hpge_mean': 0.0,
                'hpge_std': 1.0,
                'lyso_max': 1.0,
                'lyso_min': 0.0,
                'hpge_max': 1.0,
                'hpge_min': 0.0
            }
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 生成缓存键（包含数据集类型以避免混淆）
        cache_key = f"{self.dataset_type}_{idx}"
        
        # 检查缓存并获取数据的副本
        with self._cache_lock:
            if cache_key in self._cache:
                # 创建数据副本以避免竞争条件
                lyso_spectrum, hpge_spectrum = self._cache[cache_key]
                lyso_spectrum = lyso_spectrum.copy()
                hpge_spectrum = hpge_spectrum.copy()
            else:
                # 加载数据
                lyso_spectrum, hpge_spectrum = self._load_file(idx)
                
                if lyso_spectrum is None or hpge_spectrum is None:
                    raise ValueError(f"Failed to load data from file index {idx}")
                
                # 更新缓存
                if len(self._cache) >= self.cache_size:
                    # FIFO移除最早的项
                    self._cache.popitem(last=False)
                    
                # 存储数据副本到缓存
                self._cache[cache_key] = (lyso_spectrum.copy(), hpge_spectrum.copy())
        
        # 使用统一的归一化器
        lyso_spectrum, hpge_spectrum = self.normalizer.normalize(lyso_spectrum, hpge_spectrum)
        
        # 转换为张量
        lyso_tensor = torch.from_numpy(lyso_spectrum.astype(np.float32)).unsqueeze(0)
        hpge_tensor = torch.from_numpy(hpge_spectrum.astype(np.float32)).unsqueeze(0)
        
        # 应用额外的变换
        if self.transform:
            lyso_tensor = self.transform(lyso_tensor)
            hpge_tensor = self.transform(hpge_tensor)
        
        return lyso_tensor, hpge_tensor
    
    def _load_file(self, idx):
        """加载单个文件，包含错误处理和验证"""
        file_path = Path(self.file_list[idx])
        
        try:
            if file_path.suffix == '.h5':
                with h5py.File(file_path, 'r') as f:
                    lyso_spectrum = f['lyso'][:]
                    hpge_spectrum = f['hpge'][:]
            elif file_path.suffix == '.npy':
                data = np.load(file_path)
                if isinstance(data, np.ndarray) and len(data.shape) == 2:
                    lyso_spectrum = data[0]
                    hpge_spectrum = data[1]
                else:
                    raise ValueError(f"Invalid data shape in {file_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # 验证数据维度
            expected_length = 4096  # 预期的光谱长度
            if len(lyso_spectrum) != expected_length or len(hpge_spectrum) != expected_length:
                print(f"Warning: Unexpected spectrum length in {file_path}: "
                      f"lyso={len(lyso_spectrum)}, hpge={len(hpge_spectrum)}")
                
                # 调整到预期长度
                lyso_spectrum = self._adjust_spectrum_length(lyso_spectrum, expected_length)
                hpge_spectrum = self._adjust_spectrum_length(hpge_spectrum, expected_length)
            
            return lyso_spectrum, hpge_spectrum
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # 不返回None，而是抛出异常
            raise
    
    def _adjust_spectrum_length(self, spectrum, target_length):
        """调整光谱长度到目标长度"""
        current_length = len(spectrum)
        
        if current_length == target_length:
            return spectrum
        elif current_length > target_length:
            # 截断
            return spectrum[:target_length]
        else:
            # 填充
            padded = np.zeros(target_length, dtype=spectrum.dtype)
            padded[:current_length] = spectrum
            return padded


def create_isolated_data_loaders(train_path, val_path, batch_size, num_workers=4, 
                                 cache_size=100, **kwargs):
    """
    创建隔离的数据加载器，确保训练集和验证集缓存分离
    """
    # 创建数据集
    train_dataset = IsolatedSpectralDataset(
        train_path,
        cache_size=cache_size,
        dataset_type='train',
        **kwargs
    )
    
    val_dataset = IsolatedSpectralDataset(
        val_path,
        cache_size=cache_size,
        dataset_type='val',
        **kwargs
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,  # 验证使用较少的工作进程
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader