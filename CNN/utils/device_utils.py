import torch
import os
from typing import Union, Optional


class DeviceManager:
    """
    统一的设备管理器，解决设备ID验证逻辑错误
    """
    
    @staticmethod
    def get_device(device_id: Optional[Union[int, str]] = None) -> torch.device:
        """
        获取训练设备，正确处理各种设备ID情况
        
        Args:
            device_id: 设备ID，可以是：
                - None: 自动选择
                - -1: 自动选择
                - 'cpu': 使用CPU
                - 'cuda': 使用默认GPU
                - 0,1,2...: 指定GPU ID
                - 'cuda:0', 'cuda:1': 指定GPU
                
        Returns:
            torch.device对象
        """
        # CPU情况
        if device_id == 'cpu':
            return torch.device('cpu')
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            return torch.device('cpu')
        
        # 处理各种设备ID情况
        if device_id is None or device_id == -1 or device_id == 'cuda':
            # 自动选择设备
            return torch.device('cuda')
        
        # 处理字符串格式的设备ID
        if isinstance(device_id, str):
            if device_id.startswith('cuda:'):
                try:
                    gpu_id = int(device_id.split(':')[1])
                    return DeviceManager._validate_gpu_id(gpu_id)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid device string: {device_id}")
            else:
                raise ValueError(f"Unknown device string: {device_id}")
        
        # 处理整数设备ID
        if isinstance(device_id, int):
            if device_id < -1:
                raise ValueError(f"Invalid device ID: {device_id}. Must be >= -1")
            elif device_id == -1:
                return torch.device('cuda')
            else:
                return DeviceManager._validate_gpu_id(device_id)
        
        raise ValueError(f"Invalid device_id type: {type(device_id)}")
    
    @staticmethod
    def _validate_gpu_id(gpu_id: int) -> torch.device:
        """验证GPU ID是否有效"""
        device_count = torch.cuda.device_count()
        
        if gpu_id >= device_count:
            raise ValueError(
                f"Invalid GPU ID {gpu_id}. "
                f"Available GPUs: 0-{device_count-1} (total: {device_count})"
            )
        
        return torch.device(f'cuda:{gpu_id}')
    
    @staticmethod
    def set_device(device: torch.device, gpu_id: Optional[int] = None):
        """
        设置当前设备
        
        Args:
            device: torch设备对象
            gpu_id: 可选的GPU ID（用于分布式训练）
        """
        if device.type == 'cuda':
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
            else:
                # 从设备字符串提取GPU ID
                if device.index is not None:
                    torch.cuda.set_device(device.index)
    
    @staticmethod
    def get_device_info(device: Optional[torch.device] = None) -> dict:
        """
        获取设备信息
        
        Args:
            device: 设备对象，None则获取当前设备
            
        Returns:
            设备信息字典
        """
        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        
        info = {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available() and device != 'cpu':
            device_id = device.index if hasattr(device, 'index') else 0
            if device_id is not None:
                info.update({
                    'device_name': torch.cuda.get_device_name(device_id),
                    'device_capability': torch.cuda.get_device_capability(device_id),
                    'total_memory': torch.cuda.get_device_properties(device_id).total_memory,
                    'allocated_memory': torch.cuda.memory_allocated(device_id),
                    'cached_memory': torch.cuda.memory_reserved(device_id)
                })
        
        return info
    
    @staticmethod
    def auto_select_device(prefer_gpu: bool = True, 
                          min_memory: int = 1024 * 1024 * 1024) -> torch.device:
        """
        自动选择最佳设备
        
        Args:
            prefer_gpu: 是否优先选择GPU
            min_memory: 最小所需内存（字节）
            
        Returns:
            选中的设备
        """
        if not prefer_gpu or not torch.cuda.is_available():
            return torch.device('cpu')
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return torch.device('cpu')
        
        # 选择有最多可用内存的GPU
        best_device = 0
        max_free_memory = 0
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free_memory = props.total_memory - allocated
            
            if free_memory > max_free_memory and free_memory >= min_memory:
                max_free_memory = free_memory
                best_device = i
        
        if max_free_memory < min_memory:
            print(f"No GPU with sufficient memory ({min_memory} bytes), using CPU")
            return torch.device('cpu')
        
        return torch.device(f'cuda:{best_device}')


# 便捷函数
def get_device(device_id=None):
    """获取设备的便捷函数"""
    return DeviceManager.get_device(device_id)


def print_device_info(device=None):
    """打印设备信息"""
    info = DeviceManager.get_device_info(device)
    print("Device Information:")
    print("-" * 50)
    for key, value in info.items():
        if 'memory' in key and isinstance(value, (int, float)):
            # 转换内存为人类可读格式
            value = f"{value / (1024**3):.2f} GB"
        print(f"  {key}: {value}")
    print("-" * 50)