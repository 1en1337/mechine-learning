import numpy as np
import h5py
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import lmdb
import torch
from tqdm import tqdm
import shutil
import argparse


class DataPreprocessor:
    """数据预处理和缓存工具"""
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
    
    def preprocess_spectrum(self, spectrum):
        """预处理单个能谱"""
        # 归一化
        spectrum = spectrum.astype(np.float32)
        spectrum = spectrum / (np.max(spectrum) + 1e-8)
        
        # 可以添加更多预处理步骤
        # 例如：平滑、去噪等
        
        return spectrum
    
    def process_file(self, file_path):
        """处理单个文件"""
        try:
            if str(file_path).endswith('.h5'):
                with h5py.File(file_path, 'r') as f:
                    lyso = f['lyso'][:]
                    hpge = f['hpge'][:]
            else:
                data = np.load(file_path)
                lyso = data['lyso']
                hpge = data['hpge']
            
            # 预处理
            lyso = self.preprocess_spectrum(lyso)
            hpge = self.preprocess_spectrum(hpge)
            
            return lyso, hpge, str(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None, None
    
    def create_lmdb_cache(self, source_dir, lmdb_path, map_size=1e12):
        """创建LMDB缓存数据库"""
        source_path = Path(source_dir)
        files = list(source_path.rglob('*.h5')) + list(source_path.rglob('*.npy'))
        
        print(f"Found {len(files)} files to process")
        
        # 创建LMDB环境
        env = lmdb.open(str(lmdb_path), map_size=int(map_size))
        
        # 并行处理文件
        processed_count = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(self.process_file, f): i 
                      for i, f in enumerate(files)}
            
            # 处理结果
            with env.begin(write=True) as txn:
                for future in tqdm(as_completed(futures), total=len(files)):
                    lyso, hpge, file_path = future.result()
                    
                    if lyso is not None and hpge is not None:
                        idx = futures[future]
                        
                        # 存储到LMDB
                        lyso_key = f'lyso_{idx}'.encode()
                        hpge_key = f'hpge_{idx}'.encode()
                        path_key = f'path_{idx}'.encode()
                        
                        txn.put(lyso_key, lyso.tobytes())
                        txn.put(hpge_key, hpge.tobytes())
                        txn.put(path_key, file_path.encode())
                        
                        processed_count += 1
        
        # 存储元数据
        with env.begin(write=True) as txn:
            txn.put(b'__len__', str(processed_count).encode())
            txn.put(b'__shape__', str(lyso.shape).encode())
        
        env.close()
        print(f"Processed {processed_count} files into LMDB cache at {lmdb_path}")
    
    def create_memory_mapped_cache(self, source_dir, cache_dir):
        """创建内存映射缓存文件"""
        source_path = Path(source_dir)
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        files = list(source_path.rglob('*.h5')) + list(source_path.rglob('*.npy'))
        
        # 创建大的内存映射文件
        n_samples = len(files)
        n_channels = 4096
        
        lyso_mmap_path = cache_path / 'lyso_cache.dat'
        hpge_mmap_path = cache_path / 'hpge_cache.dat'
        
        # 创建内存映射数组
        lyso_mmap = np.memmap(lyso_mmap_path, dtype='float32', mode='w+', 
                             shape=(n_samples, n_channels))
        hpge_mmap = np.memmap(hpge_mmap_path, dtype='float32', mode='w+', 
                             shape=(n_samples, n_channels))
        
        try:
            # 并行处理并写入
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self.process_file, f): i 
                          for i, f in enumerate(files)}
                
                for future in tqdm(as_completed(futures), total=len(files)):
                    lyso, hpge, _ = future.result()
                    
                    if lyso is not None and hpge is not None:
                        idx = futures[future]
                        lyso_mmap[idx] = lyso
                        hpge_mmap[idx] = hpge
        finally:
            # 确保正确刷新和释放内存映射
            if 'lyso_mmap' in locals():
                lyso_mmap.flush()  # 显式刷新到磁盘
                del lyso_mmap
            if 'hpge_mmap' in locals():
                hpge_mmap.flush()  # 显式刷新到磁盘
                del hpge_mmap
            
            # Windows上可能需要额外的垃圾回收
            import gc
            gc.collect()
        
        # 保存索引信息
        index_info = {
            'n_samples': n_samples,
            'n_channels': n_channels,
            'file_paths': [str(f) for f in files]
        }
        np.save(cache_path / 'index_info.npy', index_info)
        
        print(f"Created memory-mapped cache at {cache_path}")
    
    def create_sharded_dataset(self, source_dir, output_dir, shard_size=10000):
        """创建分片数据集，适合超大规模数据"""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(source_path.rglob('*.h5')) + list(source_path.rglob('*.npy'))
        n_shards = (len(files) + shard_size - 1) // shard_size
        
        print(f"Creating {n_shards} shards from {len(files)} files")
        
        for shard_idx in range(n_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(files))
            shard_files = files[start_idx:end_idx]
            
            # 处理当前分片
            lyso_data = []
            hpge_data = []
            
            for file_path in tqdm(shard_files, desc=f"Shard {shard_idx}"):
                lyso, hpge, _ = self.process_file(file_path)
                if lyso is not None and hpge is not None:
                    lyso_data.append(lyso)
                    hpge_data.append(hpge)
            
            # 保存分片
            shard_path = output_path / f'shard_{shard_idx:04d}.npz'
            np.savez_compressed(
                shard_path,
                lyso=np.array(lyso_data),
                hpge=np.array(hpge_data)
            )
            
            print(f"Saved shard {shard_idx} with {len(lyso_data)} samples")


def optimize_data_pipeline(config):
    """优化数据处理管道的建议配置"""
    recommendations = {
        'data_format': 'lmdb',  # 推荐使用LMDB格式
        'batch_size': 64,       # 适中的批次大小
        'num_workers': min(16, mp.cpu_count()),  # 多进程加载
        'pin_memory': True,     # GPU内存固定
        'prefetch_factor': 2,   # 预取因子
        'persistent_workers': True,  # 保持worker存活
        'shuffle_buffer_size': 10000,  # 打乱缓冲区大小
        'cache_size': '100GB',  # 缓存大小建议
    }
    
    # 根据数据规模调整
    if config.get('data_size', 0) > 1000000:  # 百万级数据
        recommendations.update({
            'use_distributed': True,
            'num_gpus': torch.cuda.device_count(),
            'gradient_accumulation_steps': 4,
            'mixed_precision': True,  # 使用混合精度训练
        })
    
    return recommendations


def main():
    # Windows multiprocessing 支持
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='Preprocess spectral data')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Source directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--format', choices=['lmdb', 'mmap', 'shard'], 
                       default='lmdb', help='Output format')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--shard_size', type=int, default=10000,
                       help='Number of samples per shard')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(num_workers=args.num_workers)
    
    if args.format == 'lmdb':
        preprocessor.create_lmdb_cache(args.source_dir, args.output_dir)
    elif args.format == 'mmap':
        preprocessor.create_memory_mapped_cache(args.source_dir, args.output_dir)
    elif args.format == 'shard':
        preprocessor.create_sharded_dataset(args.source_dir, args.output_dir, 
                                          args.shard_size)


if __name__ == '__main__':
    main()