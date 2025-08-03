import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import h5py
import os
import random
from pathlib import Path
import threading
from collections import OrderedDict
import concurrent.futures
import pickle
from .normalization import SpectralNormalizer, NormalizationMethod


class SpectralDataset(Dataset):
    def __init__(self, data_path, transform=None, normalize=True, 
                 normalization_method='max', index_cache_path=None, 
                 use_mmap=True, cache_size=100):
        self.data_path = Path(data_path)
        self.transform = transform
        self.use_mmap = use_mmap
        self.cache_size = cache_size
        self.index_cache_path = index_cache_path or self.data_path / '.index_cache.pkl'
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        # 数据缓存系统
        self._cache = OrderedDict()  # 使用OrderedDict显式管理顺序
        self._cache_lock = threading.Lock()
        
        # 构建文件索引
        self._build_index()
    
    def _build_index(self):
        """构建文件索引，支持缓存和验证"""
        if self.index_cache_path.exists():
            print(f"Loading index from cache: {self.index_cache_path}")
            try:
                with open(self.index_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # 验证缓存数据结构
                    if isinstance(cached_data, list) and len(cached_data) > 0:
                        # 将字符串路径转换回Path对象
                        self.file_list = [Path(f) if isinstance(f, str) else f for f in cached_data]
                        # 验证至少第一个文件存在
                        if self.file_list[0].exists():
                            print(f"Successfully loaded {len(self.file_list)} files from cache")
                            return
                        else:
                            print(f"Cached files no longer exist, rebuilding index...")
                    else:
                        print(f"Invalid cache format, rebuilding index...")
            except (pickle.UnpicklingError, EOFError, IOError, TypeError, ValueError) as e:
                print(f"Cache loading failed ({e}), rebuilding index...")
                # 删除损坏的缓存文件
                try:
                    self.index_cache_path.unlink()
                except:
                    pass
            
            self._build_fresh_index()
        else:
            self._build_fresh_index()
    
    def _build_fresh_index(self):
        """重新构建文件索引"""
        print("Building file index...")
        self.file_list = []
        
        # 递归搜索所有H5和NPY文件
        for file_path in self.data_path.rglob('*.h5'):
            self.file_list.append(file_path)
        for file_path in self.data_path.rglob('*.npy'):
            self.file_list.append(file_path)
        
        print(f"Found {len(self.file_list)} files")
        
        if len(self.file_list) == 0:
            print(f"Warning: No data files found in {self.data_path}")
            return
        
        # 保存索引缓存 - 使用字符串路径以确保可序列化
        try:
            # 确保缓存目录存在
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_cache_path, 'wb') as f:
                # 保存为字符串路径列表
                file_list_str = [str(f) for f in self.file_list]
                pickle.dump(file_list_str, f)
            print(f"Index cache saved to {self.index_cache_path}")
        except Exception as e:
            print(f"Failed to save index cache: {e}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 检查缓存并获取数据的副本以避免竞争条件
        with self._cache_lock:
            if idx in self._cache:
                # 创建数据副本以避免在锁释放后的竞争条件
                lyso_spectrum, hpge_spectrum = self._cache[idx]
                lyso_spectrum = lyso_spectrum.copy()
                hpge_spectrum = hpge_spectrum.copy()
            else:
                # 加载数据
                lyso_spectrum, hpge_spectrum = self._load_file(idx)
                
                # 更新缓存
                if len(self._cache) >= self.cache_size:
                    # 移除最早的缓存项 (FIFO)
                    self._cache.popitem(last=False)  # 移除最早的项
                # 存储数据副本到缓存
                self._cache[idx] = (lyso_spectrum.copy(), hpge_spectrum.copy())
        
        # 转换为张量 - 现在在锁外部是安全的
        lyso_tensor = torch.from_numpy(lyso_spectrum).unsqueeze(0)
        hpge_tensor = torch.from_numpy(hpge_spectrum).unsqueeze(0)
        
        if self.transform:
            lyso_tensor = self.transform(lyso_tensor)
            hpge_tensor = self.transform(hpge_tensor)
        
        return lyso_tensor, hpge_tensor
    
    def _load_file(self, idx):
        """加载单个文件，使用统一加载器"""
        file_path = self.file_list[idx]
        
        try:
            # 使用统一加载器处理不同格式
            from .unified_loader import load_spectrum_file
            lyso_spectrum, hpge_spectrum = load_spectrum_file(file_path, expected_length=4096)
            
            lyso_spectrum = lyso_spectrum.astype(np.float32)
            hpge_spectrum = hpge_spectrum.astype(np.float32)
            
            # 使用统一的归一化器
            lyso_spectrum, hpge_spectrum = self.normalizer.normalize(lyso_spectrum, hpge_spectrum)
            
            return lyso_spectrum, hpge_spectrum
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 抛出异常，让DataLoader处理
            raise IOError(f"Failed to load {file_path}: {e}") from e


class StreamingSpectralDataset(IterableDataset):
    """流式光谱数据集，适合超大规模H5/NPY数据"""
    
    def __init__(self, data_path, transform=None, normalize=True, 
                 normalization_method='max', buffer_size=1000, shuffle=True, cycle=True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.cycle = cycle
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        # 获取文件列表
        self.file_list = []
        for file_path in self.data_path.rglob('*.h5'):
            self.file_list.append(file_path)
        for file_path in self.data_path.rglob('*.npy'):
            self.file_list.append(file_path)
        
        if not self.file_list:
            raise ValueError(f"No H5/NPY files found in {data_path}")
        
        print(f"Streaming dataset contains {len(self.file_list)} files")
    
    def __iter__(self):
        """流式迭代器"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 分配文件给不同的worker
        if worker_info is None:
            file_list = self.file_list
        else:
            per_worker = len(self.file_list) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else len(self.file_list)
            file_list = self.file_list[start_idx:end_idx]
        
        while True:
            if self.shuffle:
                random.shuffle(file_list)
            
            for file_path in file_list:
                try:
                    if file_path.suffix == '.h5':
                        with h5py.File(file_path, 'r') as f:
                            lyso = f['lyso'][:]
                            hpge = f['hpge'][:]
                    else:
                        data = np.load(file_path)
                        lyso = data['lyso']
                        hpge = data['hpge']
                    
                    lyso = lyso.astype(np.float32)
                    hpge = hpge.astype(np.float32)
                    
                    # 使用统一的归一化器
                    lyso, hpge = self.normalizer.normalize(lyso, hpge)
                    
                    lyso_tensor = torch.from_numpy(lyso).unsqueeze(0)
                    hpge_tensor = torch.from_numpy(hpge).unsqueeze(0)
                    
                    if self.transform:
                        lyso_tensor = self.transform(lyso_tensor)
                        hpge_tensor = self.transform(hpge_tensor)
                    
                    yield lyso_tensor, hpge_tensor
                    
                except Exception as e:
                    # 不是静默地继续，而是记录错误并抛出
                    error_msg = f"Error loading {file_path}: {e}"
                    print(error_msg)
                    # 将错误信息保存到日志文件
                    if hasattr(self, '_error_log'):
                        self._error_log.append(error_msg)
                    # 如果错误过多，抛出异常
                    if hasattr(self, '_error_count'):
                        self._error_count += 1
                        if self._error_count > 10:  # 容忍10个错误
                            raise RuntimeError(f"Too many data loading errors ({self._error_count})")
                    continue
            
            if not self.cycle:
                # 在分布式环境中，需要确保所有worker都完成
                # 通过返回而不是break来结束迭代
                return


class ErrorHandlingDataLoader(DataLoader):
    """自定义DataLoader，能够跳过错误的样本"""
    def __iter__(self):
        iterator = super().__iter__()
        while True:
            try:
                batch = next(iterator)
                yield batch
            except StopIteration:
                break
            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                continue


def create_data_loaders(train_path, val_path, batch_size=32, num_workers=4, 
                       use_streaming=False, **kwargs):
    """创建优化的数据加载器，支持大规模H5/NPY数据
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        use_streaming: 是否使用流式加载（适合超大数据集）
        **kwargs: 其他参数传递给SpectralDataset
    """
    # 参数验证
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training path not found: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Validation path not found: {val_path}")
    if use_streaming:
        # 使用流式数据集（适合超大规模数据）
        train_dataset = StreamingSpectralDataset(train_path, **kwargs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # 使用优化的标准数据集
        train_dataset = SpectralDataset(train_path, **kwargs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    # 验证集始终使用标准数据集
    val_dataset = SpectralDataset(val_path, **kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_large_data_loaders(train_path, val_path, batch_size=32, num_workers=8,
                             cache_size=200, use_mmap=True):
    """创建适合大规模数据的高性能加载器"""
    train_dataset = SpectralDataset(
        train_path, 
        cache_size=cache_size,
        use_mmap=use_mmap
    )
    val_dataset = SpectralDataset(
        val_path,
        cache_size=cache_size//2,
        use_mmap=use_mmap
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True
    )
    
    return train_loader, val_loader