import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
import asyncio
import aiofiles
import concurrent.futures
from collections import deque
import threading
import queue
import time
from .normalization import SpectralNormalizer, NormalizationMethod


class AsyncSpectralDataset(Dataset):
    """
    异步光谱数据集，使用预取和并行I/O优化性能
    """
    def __init__(self, data_path, cache_size=200, prefetch_size=50, 
                 num_io_workers=4, transform=None, normalize=True,
                 normalization_method='mean_std'):
        self.data_path = Path(data_path)
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.num_io_workers = num_io_workers
        self.transform = transform
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        # 构建文件列表
        self.file_list = self._build_file_list()
        
        # 缓存系统
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_queue = deque(maxlen=cache_size)
        
        # 预取队列
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.prefetch_indices = set()
        self.prefetch_lock = threading.Lock()
        
        # 线程控制
        self._stop_event = threading.Event()
        self._stopped = False
        
        # I/O线程池
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_io_workers
        )
        
        # 启动预取线程
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        # 计算并设置统计信息
        if self.normalizer.method == NormalizationMethod.MEAN_STD:
            self._compute_and_set_stats()
        
    def _build_file_list(self):
        """构建文件列表"""
        file_list = []
        for pattern in ['*.h5', '*.npy']:
            file_list.extend(self.data_path.rglob(pattern))
        
        if not file_list:
            raise ValueError(f"No data files found in {self.data_path}")
        
        return sorted([str(f) for f in file_list])
    
    def _compute_and_set_stats(self):
        """计算并设置数据集统计信息"""
        sample_size = min(100, len(self.file_list))
        indices = np.random.choice(len(self.file_list), sample_size, replace=False)
        
        samples = []
        for idx in indices:
            try:
                lyso, hpge = self._load_file_sync(idx)
                samples.append((lyso, hpge))
            except Exception:
                continue
        
        if samples:
            self.normalizer.compute_stats(samples)
    
    def _load_file_sync(self, idx):
        """同步加载文件（用于初始化）"""
        file_path = Path(self.file_list[idx])
        
        if file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                lyso_spectrum = f['lyso'][:]
                hpge_spectrum = f['hpge'][:]
        elif file_path.suffix == '.npy':
            data = np.load(file_path)
            lyso_spectrum = data[0]
            hpge_spectrum = data[1]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 验证和调整长度 - 使用全局配置
        from .constants import get_spectrum_length
        expected_length = get_spectrum_length()
        lyso_spectrum = self._adjust_length(lyso_spectrum, expected_length)
        hpge_spectrum = self._adjust_length(hpge_spectrum, expected_length)
        
        return lyso_spectrum, hpge_spectrum
    
    def _adjust_length(self, spectrum, target_length):
        """调整光谱长度"""
        if len(spectrum) == target_length:
            return spectrum
        elif len(spectrum) > target_length:
            return spectrum[:target_length]
        else:
            padded = np.zeros(target_length, dtype=spectrum.dtype)
            padded[:len(spectrum)] = spectrum
            return padded
    
    def _load_file_async(self, idx):
        """异步加载文件"""
        future = self.io_executor.submit(self._load_file_sync, idx)
        return future
    
    def _prefetch_worker(self):
        """预取工作线程"""
        while not self._stop_event.is_set():
            try:
                # 获取需要预取的索引
                indices_to_prefetch = []
                
                with self.prefetch_lock:
                    # 查找未在缓存中的索引
                    for i in range(len(self.file_list)):
                        if i not in self.cache and i not in self.prefetch_indices:
                            indices_to_prefetch.append(i)
                            self.prefetch_indices.add(i)
                            
                            if len(indices_to_prefetch) >= self.prefetch_size // 2:
                                break
                
                # 异步加载
                if indices_to_prefetch:
                    futures = {
                        idx: self._load_file_async(idx) 
                        for idx in indices_to_prefetch
                    }
                    
                    # 等待完成
                    for idx, future in futures.items():
                        try:
                            lyso, hpge = future.result(timeout=5.0)
                            
                            # 添加到缓存
                            with self.cache_lock:
                                if len(self.cache) >= self.cache_size:
                                    # 移除最旧的项
                                    if self.cache_queue:
                                        old_idx = self.cache_queue.popleft()
                                        self.cache.pop(old_idx, None)
                                
                                self.cache[idx] = (lyso.copy(), hpge.copy())
                                self.cache_queue.append(idx)
                            
                            # 从预取集合中移除
                            with self.prefetch_lock:
                                self.prefetch_indices.discard(idx)
                                
                        except Exception as e:
                            print(f"Prefetch error for index {idx}: {e}")
                            with self.prefetch_lock:
                                self.prefetch_indices.discard(idx)
                
                # 短暂休眠，同时检查停止信号
                if self._stop_event.wait(0.01):
                    break
                
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Prefetch worker error: {e}")
                    if self._stop_event.wait(0.1):
                        break
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 检查缓存
        with self.cache_lock:
            if idx in self.cache:
                lyso_spectrum, hpge_spectrum = self.cache[idx]
                lyso_spectrum = lyso_spectrum.copy()
                hpge_spectrum = hpge_spectrum.copy()
                hit = True
            else:
                hit = False
        
        # 缓存未命中，同步加载
        if not hit:
            lyso_spectrum, hpge_spectrum = self._load_file_sync(idx)
            
            # 更新缓存
            with self.cache_lock:
                if len(self.cache) >= self.cache_size:
                    if self.cache_queue:
                        old_idx = self.cache_queue.popleft()
                        self.cache.pop(old_idx, None)
                
                self.cache[idx] = (lyso_spectrum.copy(), hpge_spectrum.copy())
                self.cache_queue.append(idx)
        
        # 使用统一的归一化器
        lyso_spectrum, hpge_spectrum = self.normalizer.normalize(lyso_spectrum, hpge_spectrum)
        
        # 转换为张量
        lyso_tensor = torch.from_numpy(lyso_spectrum.astype(np.float32)).unsqueeze(0)
        hpge_tensor = torch.from_numpy(hpge_spectrum.astype(np.float32)).unsqueeze(0)
        
        # 应用变换
        if self.transform:
            lyso_tensor = self.transform(lyso_tensor)
            hpge_tensor = self.transform(hpge_tensor)
        
        return lyso_tensor, hpge_tensor
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
        return False
    
    def close(self):
        """正确关闭数据集并清理资源"""
        if self._stopped:
            return
            
        self._stopped = True
        
        # 停止预取线程
        self._stop_event.set()
        
        # 等待预取线程结束
        if hasattr(self, 'prefetch_thread') and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
            if self.prefetch_thread.is_alive():
                print("Warning: Prefetch thread did not stop cleanly")
        
        # 关闭I/O线程池
        if hasattr(self, 'io_executor'):
            self.io_executor.shutdown(wait=True, cancel_futures=True)
        
        # 清理缓存
        with self.cache_lock:
            self.cache.clear()
        self.cache_queue.clear()
        self.prefetch_indices.clear()
    
    def __del__(self):
        """清理资源"""
        try:
            self.close()
        except:
            pass  # 忽略清理时的错误


class OptimizedDataLoader:
    """
    优化的数据加载器包装器
    """
    @staticmethod
    def create_loaders(train_path, val_path, batch_size, num_workers=4, 
                      cache_size=200, pin_memory=True, **kwargs):
        """
        创建优化的数据加载器
        """
        # 创建异步数据集
        train_dataset = AsyncSpectralDataset(
            train_path,
            cache_size=cache_size,
            prefetch_size=batch_size * 2,
            num_io_workers=max(2, num_workers // 2),
            **kwargs
        )
        
        val_dataset = AsyncSpectralDataset(
            val_path,
            cache_size=cache_size // 2,
            prefetch_size=batch_size,
            num_io_workers=max(2, num_workers // 4),
            **kwargs
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True  # 保证批次大小一致
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers // 2,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 1),
            drop_last=False
        )
        
        return train_loader, val_loader