import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import h5py
import os
from pathlib import Path
import random
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import mmap
import lmdb
from .normalization import SpectralNormalizer, NormalizationMethod


class LargeSpectralDataset(Dataset):
    """优化版Dataset，支持百万级数据量"""
    
    def __init__(self, data_path, transform=None, normalize=True, 
                 normalization_method='max', index_cache_path=None, use_mmap=True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.use_mmap = use_mmap
        self.index_cache_path = index_cache_path or self.data_path / '.index_cache.pkl'
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        self._build_index()
        
    def _build_index(self):
        """构建文件索引，支持缓存"""
        if self.index_cache_path.exists():
            print(f"Loading index from cache: {self.index_cache_path}")
            with open(self.index_cache_path, 'rb') as f:
                self.file_index = pickle.load(f)
        else:
            print("Building file index...")
            self.file_index = []
            
            for file_path in self.data_path.rglob('*.h5'):
                self.file_index.append(str(file_path))
            for file_path in self.data_path.rglob('*.npy'):
                self.file_index.append(str(file_path))
            
            print(f"Found {len(self.file_index)} files")
            
            # 保存索引缓存
            # 确保缓存目录存在
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_cache_path, 'wb') as f:
                pickle.dump(self.file_index, f)
    
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        file_path = self.file_index[idx]
        
        try:
            if file_path.endswith('.h5'):
                with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
                    lyso_spectrum = f['lyso'][:]
                    hpge_spectrum = f['hpge'][:]
            else:
                if self.use_mmap:
                    # 使用内存映射加速
                    data = np.load(file_path, mmap_mode='r')
                else:
                    data = np.load(file_path)
                lyso_spectrum = data['lyso']
                hpge_spectrum = data['hpge']
            
            lyso_spectrum = lyso_spectrum.astype(np.float32)
            hpge_spectrum = hpge_spectrum.astype(np.float32)
            
            # 使用统一的归一化器
            lyso_spectrum, hpge_spectrum = self.normalizer.normalize(lyso_spectrum, hpge_spectrum)
            
            lyso_tensor = torch.from_numpy(lyso_spectrum).unsqueeze(0)
            hpge_tensor = torch.from_numpy(hpge_spectrum).unsqueeze(0)
            
            if self.transform:
                lyso_tensor = self.transform(lyso_tensor)
                hpge_tensor = self.transform(hpge_tensor)
            
            return lyso_tensor, hpge_tensor
            
        except (OSError, IOError, h5py._hl.files.FileID) as e:
            print(f"Error loading {file_path}: {e}")
            # 返回零张量避免训练中断
            return torch.zeros(1, 4096), torch.zeros(1, 4096)


class StreamingSpectralDataset(IterableDataset):
    """流式数据集，适合超大规模数据"""
    
    def __init__(self, data_path, transform=None, normalize=True, 
                 normalization_method='max', buffer_size=1000, num_workers=4):
        self.data_path = Path(data_path)
        self.transform = transform
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 获取所有文件列表
        file_list = []
        for pattern in ['*.h5', '*.npy']:
            file_list.extend(self.data_path.rglob(pattern))
        
        # 分配给当前worker的文件
        if worker_info is None:
            files = file_list
        else:
            per_worker = len(file_list) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(file_list)
            files = file_list[start:end]
        
        # 随机打乱
        random.shuffle(files)
        
        # 流式读取
        for file_path in files:
            try:
                if str(file_path).endswith('.h5'):
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
                
                yield lyso_tensor, hpge_tensor
                
            except Exception as e:
                continue


class LMDBSpectralDataset(Dataset):
    """使用LMDB数据库存储，极大提升读取速度"""
    
    def __init__(self, lmdb_path, transform=None, normalize=True, normalization_method='max'):
        self.lmdb_path = lmdb_path
        self.transform = transform
        
        # 设置归一化器
        if normalize:
            self.normalizer = SpectralNormalizer(normalization_method)
        else:
            self.normalizer = SpectralNormalizer(NormalizationMethod.NONE)
        
        # 打开LMDB环境
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False,
                            readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            # 获取总条目数
            total_entries = txn.stat()['entries']
            # 检查是否有__len__键
            len_bytes = txn.get(b'__len__')
            if len_bytes:
                self.length = int(len_bytes.decode())
                # 验证长度是否合理
                if self.length * 2 > total_entries:
                    print(f"Warning: Stored length ({self.length}) seems incorrect based on total entries ({total_entries}).")
                    # 使用估算值，减去元数据条目
                    self.length = (total_entries - 1) // 2  # -1 for __len__ key
            else:
                # 如果没有__len__，假设每个样本有2个条目（lyso和hpge）
                print("Warning: No __len__ key found in LMDB. Using estimated length.")
                self.length = total_entries // 2
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            lyso_key = f'lyso_{idx}'.encode()
            hpge_key = f'hpge_{idx}'.encode()
            
            lyso_bytes = txn.get(lyso_key)
            hpge_bytes = txn.get(hpge_key)
            
            if lyso_bytes is None or hpge_bytes is None:
                raise IndexError(f"Sample {idx} not found in LMDB database. Max index is {self.length - 1}")
            
            lyso_spectrum = np.frombuffer(lyso_bytes, dtype=np.float32)
            hpge_spectrum = np.frombuffer(hpge_bytes, dtype=np.float32)
        
        # 使用统一的归一化器
        lyso_spectrum, hpge_spectrum = self.normalizer.normalize(lyso_spectrum, hpge_spectrum)
        
        lyso_tensor = torch.from_numpy(lyso_spectrum).unsqueeze(0)
        hpge_tensor = torch.from_numpy(hpge_spectrum).unsqueeze(0)
        
        if self.transform:
            lyso_tensor = self.transform(lyso_tensor)
            hpge_tensor = self.transform(hpge_tensor)
        
        return lyso_tensor, hpge_tensor
    
    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except:
                pass  # 忽略关闭时的错误


def create_lmdb_dataset(source_path, lmdb_path, map_size=1e12):
    """将数据转换为LMDB格式"""
    env = lmdb.open(lmdb_path, map_size=int(map_size))
    
    source_path = Path(source_path)
    files = list(source_path.rglob('*.h5')) + list(source_path.rglob('*.npy'))
    
    with env.begin(write=True) as txn:
        for idx, file_path in enumerate(files):
            if idx % 1000 == 0:
                print(f"Processing {idx}/{len(files)}")
            
            if str(file_path).endswith('.h5'):
                with h5py.File(file_path, 'r') as f:
                    lyso = f['lyso'][:].astype(np.float32)
                    hpge = f['hpge'][:].astype(np.float32)
            else:
                data = np.load(file_path)
                lyso = data['lyso'].astype(np.float32)
                hpge = data['hpge'].astype(np.float32)
            
            lyso_key = f'lyso_{idx}'.encode()
            hpge_key = f'hpge_{idx}'.encode()
            
            txn.put(lyso_key, lyso.tobytes())
            txn.put(hpge_key, hpge.tobytes())
        
        # 存储数据集长度
        txn.put(b'__len__', str(len(files)).encode())
    
    env.close()
    print(f"LMDB dataset created at {lmdb_path}")


class PrefetchDataLoader:
    """预取数据加载器，减少I/O等待"""
    
    def __init__(self, dataset, batch_size, num_workers=4, prefetch_factor=2):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False  # 保持worker进程活跃
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_large_data_loaders(train_path, val_path, batch_size=32, 
                            num_workers=8, use_lmdb=False):
    """创建适合大规模数据的加载器"""
    
    if use_lmdb:
        # 使用LMDB格式
        train_dataset = LMDBSpectralDataset(train_path)
        val_dataset = LMDBSpectralDataset(val_path)
    else:
        # 使用优化的文件读取
        train_dataset = LargeSpectralDataset(train_path)
        val_dataset = LargeSpectralDataset(val_path)
    
    train_loader = PrefetchDataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=True
    )
    
    return train_loader, val_loader