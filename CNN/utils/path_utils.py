import os
from pathlib import Path
from typing import Union, Optional
import platform


def normalize_path(path: Union[str, Path]) -> Path:
    """
    标准化路径，确保跨平台兼容性
    
    Args:
        path: 输入路径（字符串或Path对象）
        
    Returns:
        标准化的Path对象
    """
    if isinstance(path, str):
        # 替换混合的路径分隔符
        path = path.replace('\\', '/')
        path = Path(path)
    
    # 解析为绝对路径
    path = path.resolve()
    
    return path


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的Path对象
    """
    # 从当前文件向上查找，直到找到包含特定标记文件的目录
    current = Path(__file__).parent.parent  # utils的父目录
    
    # 检查一些项目标记文件
    markers = ['train.py', 'README.md', 'configs']
    
    while current.parent != current:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent
    
    # 如果找不到，返回当前文件的父目录的父目录
    return Path(__file__).parent.parent


def get_default_data_path(dataset_type: str = 'train') -> Path:
    """
    获取默认数据路径，根据平台自动调整
    
    Args:
        dataset_type: 数据集类型 ('train', 'val', 'test')
        
    Returns:
        数据路径的Path对象
    """
    project_root = get_project_root()
    
    # 优先使用相对于项目根目录的路径
    data_path = project_root / 'dataset' / dataset_type
    
    # 如果不存在，尝试环境变量
    env_var = f'CNN_DATA_{dataset_type.upper()}_PATH'
    if env_var in os.environ:
        data_path = normalize_path(os.environ[env_var])
    
    return data_path


def ensure_path_exists(path: Union[str, Path], is_dir: bool = True) -> Path:
    """
    确保路径存在，如果不存在则创建
    
    Args:
        path: 路径
        is_dir: 是否是目录（True）还是文件（False）
        
    Returns:
        Path对象
    """
    path = normalize_path(path)
    
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def convert_path_for_config(path: Union[str, Path]) -> str:
    """
    将路径转换为配置文件中的格式（使用正斜杠）
    
    Args:
        path: 输入路径
        
    Returns:
        使用正斜杠的路径字符串
    """
    path = normalize_path(path)
    # 始终使用正斜杠，即使在Windows上
    return str(path).replace('\\', '/')


def resolve_data_path(path_config: Union[str, Path, None], 
                     dataset_type: str = 'train') -> Path:
    """
    解析数据路径，优先使用配置路径，否则使用默认路径
    
    Args:
        path_config: 配置中的路径
        dataset_type: 数据集类型
        
    Returns:
        解析后的Path对象
    """
    if path_config:
        return normalize_path(path_config)
    else:
        return get_default_data_path(dataset_type)


def is_relative_to(path: Path, other: Path) -> bool:
    """
    检查一个路径是否相对于另一个路径（兼容Python < 3.9）
    
    Args:
        path: 要检查的路径
        other: 基准路径
        
    Returns:
        是否是相对路径
    """
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def get_platform_info() -> dict:
    """
    获取平台信息，用于调试路径问题
    
    Returns:
        平台信息字典
    """
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'platform': platform.platform(),
        'path_separator': os.path.sep,
        'pathsep': os.pathsep
    }


# 导出便捷函数
def fix_path(path: Union[str, Path]) -> Path:
    """
    快速修复路径的便捷函数
    
    Args:
        path: 输入路径
        
    Returns:
        修复后的Path对象
    """
    return normalize_path(path)