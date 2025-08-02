import torch
import numpy as np
from typing import Union, Tuple, Dict, Optional
from pathlib import Path
import pickle


class NormalizationMethod:
    """标准化方法枚举"""
    MAX = 'max'  # 最大值归一化 (0-1)
    MEAN_STD = 'mean_std'  # 均值标准差归一化
    MINMAX = 'minmax'  # 最小-最大归一化
    NONE = 'none'  # 不归一化


class SpectralNormalizer:
    """
    统一的光谱归一化器，避免重复归一化问题
    """
    def __init__(self, method: str = NormalizationMethod.MAX):
        """
        Args:
            method: 归一化方法
        """
        self.method = method
        self.stats = None
        self._normalized = False
        
    def compute_stats(self, data_samples: list) -> Dict[str, float]:
        """
        计算数据集统计信息
        
        Args:
            data_samples: 数据样本列表 [(lyso, hpge), ...]
            
        Returns:
            统计信息字典
        """
        if not data_samples:
            return None
            
        lyso_list = []
        hpge_list = []
        
        for lyso, hpge in data_samples:
            lyso_list.append(lyso)
            hpge_list.append(hpge)
            
        lyso_array = np.array(lyso_list)
        hpge_array = np.array(hpge_list)
        
        self.stats = {
            'lyso_mean': float(np.mean(lyso_array)),
            'lyso_std': float(np.std(lyso_array) + 1e-8),
            'lyso_max': float(np.max(lyso_array)),
            'lyso_min': float(np.min(lyso_array)),
            'hpge_mean': float(np.mean(hpge_array)),
            'hpge_std': float(np.std(hpge_array) + 1e-8),
            'hpge_max': float(np.max(hpge_array)),
            'hpge_min': float(np.min(hpge_array))
        }
        
        return self.stats
        
    def normalize(self, lyso: np.ndarray, hpge: np.ndarray, 
                 in_place: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化光谱数据
        
        Args:
            lyso: LYSO光谱
            hpge: HPGe光谱
            in_place: 是否原地修改
            
        Returns:
            归一化后的(lyso, hpge)
        """
        if self.method == NormalizationMethod.NONE:
            return lyso, hpge
            
        if not in_place:
            lyso = lyso.copy()
            hpge = hpge.copy()
            
        if self.method == NormalizationMethod.MAX:
            # 最大值归一化
            lyso_max = np.max(lyso)
            hpge_max = np.max(hpge)
            
            if lyso_max > 0:
                lyso = lyso / lyso_max
            if hpge_max > 0:
                hpge = hpge / hpge_max
                
        elif self.method == NormalizationMethod.MEAN_STD:
            # 均值标准差归一化
            if self.stats is None:
                raise ValueError("Stats not computed. Call compute_stats first.")
                
            lyso = (lyso - self.stats['lyso_mean']) / self.stats['lyso_std']
            hpge = (hpge - self.stats['hpge_mean']) / self.stats['hpge_std']
            
        elif self.method == NormalizationMethod.MINMAX:
            # 最小-最大归一化
            if self.stats is None:
                lyso_min, lyso_max = np.min(lyso), np.max(lyso)
                hpge_min, hpge_max = np.min(hpge), np.max(hpge)
            else:
                lyso_min = self.stats['lyso_min']
                lyso_max = self.stats['lyso_max']
                hpge_min = self.stats['hpge_min']
                hpge_max = self.stats['hpge_max']
                
            if lyso_max > lyso_min:
                lyso = (lyso - lyso_min) / (lyso_max - lyso_min)
            if hpge_max > hpge_min:
                hpge = (hpge - hpge_min) / (hpge_max - hpge_min)
                
        return lyso, hpge
        
    def denormalize(self, lyso: Union[np.ndarray, torch.Tensor], 
                   hpge: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        反归一化光谱数据
        
        Args:
            lyso: 归一化的LYSO光谱
            hpge: 归一化的HPGe光谱
            
        Returns:
            反归一化后的(lyso, hpge)
        """
        if self.method == NormalizationMethod.NONE:
            return lyso, hpge
            
        # 转换为numpy
        if isinstance(lyso, torch.Tensor):
            lyso = lyso.cpu().numpy()
        if isinstance(hpge, torch.Tensor):
            hpge = hpge.cpu().numpy()
            
        lyso = lyso.copy()
        hpge = hpge.copy()
        
        if self.method == NormalizationMethod.MEAN_STD:
            if self.stats is None:
                raise ValueError("Stats not available for denormalization")
                
            lyso = lyso * self.stats['lyso_std'] + self.stats['lyso_mean']
            hpge = hpge * self.stats['hpge_std'] + self.stats['hpge_mean']
            
        elif self.method == NormalizationMethod.MINMAX:
            if self.stats is None:
                raise ValueError("Stats not available for denormalization")
                
            lyso = lyso * (self.stats['lyso_max'] - self.stats['lyso_min']) + self.stats['lyso_min']
            hpge = hpge * (self.stats['hpge_max'] - self.stats['hpge_min']) + self.stats['hpge_min']
            
        elif self.method == NormalizationMethod.MAX:
            # 无法精确反归一化，因为没有保存原始最大值
            if self.stats and 'lyso_max' in self.stats:
                lyso = lyso * self.stats['lyso_max']
                hpge = hpge * self.stats['hpge_max']
            else:
                print("Warning: Cannot denormalize MAX normalization without stats")
                
        return lyso, hpge
        
    def save_stats(self, path: Union[str, Path]):
        """保存统计信息"""
        if self.stats is None:
            raise ValueError("No stats to save")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'stats': self.stats
            }, f)
            
    def load_stats(self, path: Union[str, Path]):
        """加载统计信息"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Stats file not found: {path}")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.method = data['method']
            self.stats = data['stats']
            
    @staticmethod
    def create_from_config(config: Dict) -> 'SpectralNormalizer':
        """从配置创建归一化器"""
        data_config = config.get('data', {})
        
        # 获取归一化配置
        normalize = data_config.get('normalize', True)
        if not normalize:
            return SpectralNormalizer(NormalizationMethod.NONE)
            
        # 获取归一化方法
        norm_method = data_config.get('normalization_method', NormalizationMethod.MAX)
        
        return SpectralNormalizer(norm_method)