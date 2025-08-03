import numpy as np
import h5py
from pathlib import Path
import torch
from typing import Tuple, Optional


class UnifiedDataLoader:
    """
    统一的数据加载器，解决NPY和H5文件格式假设冲突
    """
    
    @staticmethod
    def load_spectrum_pair(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载光谱对数据，自动处理不同的文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            (lyso_spectrum, hpge_spectrum) 元组
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if file_path.suffix == '.h5':
                return UnifiedDataLoader._load_h5(file_path)
            elif file_path.suffix == '.npy':
                return UnifiedDataLoader._load_npy(file_path)
            elif file_path.suffix == '.npz':
                return UnifiedDataLoader._load_npz(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")
    
    @staticmethod
    def _load_h5(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """加载H5格式文件"""
        with h5py.File(file_path, 'r') as f:
            # 检查数据键
            if 'lyso' in f and 'hpge' in f:
                lyso_spectrum = f['lyso'][:]
                hpge_spectrum = f['hpge'][:]
            elif 'input' in f and 'target' in f:
                lyso_spectrum = f['input'][:]
                hpge_spectrum = f['target'][:]
            else:
                # 尝试通过索引访问
                keys = list(f.keys())
                if len(keys) >= 2:
                    lyso_spectrum = f[keys[0]][:]
                    hpge_spectrum = f[keys[1]][:]
                else:
                    raise KeyError(f"Cannot find appropriate data keys in {file_path}")
        
        return lyso_spectrum, hpge_spectrum
    
    @staticmethod
    def _load_npy(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """加载NPY格式文件，自动检测格式"""
        data = np.load(file_path, allow_pickle=True)
        
        # 情况1：字典格式
        if isinstance(data, dict) or (isinstance(data, np.ndarray) and data.dtype == object):
            if isinstance(data, np.ndarray):
                data = data.item()
            
            if 'lyso' in data and 'hpge' in data:
                return data['lyso'], data['hpge']
            elif 'input' in data and 'target' in data:
                return data['input'], data['target']
            else:
                keys = list(data.keys())
                if len(keys) >= 2:
                    return data[keys[0]], data[keys[1]]
                else:
                    raise ValueError(f"Dictionary format but insufficient keys in {file_path}")
        
        # 情况2：2D数组格式 [2, length]
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            if data.shape[0] == 2:
                return data[0], data[1]
            elif data.shape[1] == 2:
                return data[:, 0], data[:, 1]
            else:
                # 假设前半部分是lyso，后半部分是hpge
                mid = data.shape[1] // 2
                return data[0, :mid], data[0, mid:]
        
        # 情况3：1D数组格式（假设是拼接的）
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            mid = len(data) // 2
            return data[:mid], data[mid:]
        
        else:
            raise ValueError(f"Unsupported NPY data format in {file_path}: "
                           f"type={type(data)}, shape={data.shape if hasattr(data, 'shape') else 'N/A'}")
    
    @staticmethod
    def _load_npz(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """加载NPZ压缩格式文件"""
        with np.load(file_path) as data:
            if 'lyso' in data and 'hpge' in data:
                return data['lyso'], data['hpge']
            elif 'input' in data and 'target' in data:
                return data['input'], data['target']
            else:
                keys = list(data.keys())
                if len(keys) >= 2:
                    return data[keys[0]], data[keys[1]]
                else:
                    raise ValueError(f"Insufficient data arrays in {file_path}")
    
    @staticmethod
    def validate_spectrum_length(spectrum: np.ndarray, expected_length: int = 4096) -> np.ndarray:
        """
        验证并调整光谱长度
        
        Args:
            spectrum: 输入光谱
            expected_length: 期望长度
            
        Returns:
            调整后的光谱
        """
        current_length = len(spectrum)
        
        if current_length == expected_length:
            return spectrum
        elif current_length > expected_length:
            # 截断
            return spectrum[:expected_length]
        else:
            # 填充
            padded = np.zeros(expected_length, dtype=spectrum.dtype)
            padded[:current_length] = spectrum
            return padded
    
    @staticmethod
    def save_spectrum_pair(lyso_spectrum: np.ndarray, 
                          hpge_spectrum: np.ndarray, 
                          file_path: Path,
                          format: str = 'npz') -> None:
        """
        保存光谱对数据
        
        Args:
            lyso_spectrum: LYSO光谱
            hpge_spectrum: HPGe光谱
            file_path: 保存路径
            format: 保存格式 ('npy', 'npz', 'h5')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez_compressed(file_path, lyso=lyso_spectrum, hpge=hpge_spectrum)
        elif format == 'npy':
            # 保存为2D数组格式
            data = np.array([lyso_spectrum, hpge_spectrum])
            np.save(file_path, data)
        elif format == 'h5':
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('lyso', data=lyso_spectrum, compression='gzip')
                f.create_dataset('hpge', data=hpge_spectrum, compression='gzip')
        else:
            raise ValueError(f"Unsupported save format: {format}")


# 为了向后兼容，创建一个简单的加载函数
def load_spectrum_file(file_path: Path, expected_length: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：加载并验证光谱文件
    """
    loader = UnifiedDataLoader()
    lyso, hpge = loader.load_spectrum_pair(file_path)
    
    # 验证长度
    lyso = loader.validate_spectrum_length(lyso, expected_length)
    hpge = loader.validate_spectrum_length(hpge, expected_length)
    
    return lyso, hpge