"""
全局常量定义，确保整个项目的一致性
"""
import os
from pathlib import Path

# 光谱相关常量
SPECTRUM_LENGTH = 4096  # 标准光谱长度
MIN_SPECTRUM_LENGTH = 1024  # 最小允许的光谱长度
MAX_SPECTRUM_LENGTH = 8192  # 最大允许的光谱长度

# 数据格式常量
SUPPORTED_FILE_FORMATS = ['.h5', '.npy', '.npz']
DEFAULT_FILE_FORMAT = '.h5'

# 缓存相关常量
DEFAULT_CACHE_SIZE = 100
DEFAULT_PREFETCH_SIZE = 50
MAX_CACHE_SIZE = 1000

# 训练相关常量
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_NUM_WORKERS = 4

# 设备相关常量
DEFAULT_DEVICE = 'cuda'
MIN_GPU_MEMORY = 2 * 1024 * 1024 * 1024  # 2GB

# 损失函数相关常量
DEFAULT_PEAK_WEIGHT = 10.0
DEFAULT_BASE_WEIGHT = 1.0
DEFAULT_SMOOTHNESS_WEIGHT = 0.1
DEFAULT_FREQUENCY_WEIGHT = 0.1

# 评估相关常量
PEAK_DETECTION_THRESHOLD = 0.3  # 峰检测阈值（相对于最大值）
PEAK_DETECTION_DISTANCE = 50  # 峰之间的最小距离
FWHM_SMOOTHING_SIGMA = 2.0  # FWHM计算时的平滑参数

# 数据路径模板 - 使用相对路径或环境变量
def _get_default_data_path(dataset_type: str) -> str:
    """获取默认数据路径"""
    # 优先使用环境变量
    env_var = f'CNN_DATA_{dataset_type.upper()}_PATH'
    if env_var in os.environ:
        return os.environ[env_var]
    
    # 否则使用相对路径
    # 这里假设数据在项目根目录的dataset文件夹下
    return f'dataset/{dataset_type}'

DEFAULT_DATA_PATHS = {
    'train': _get_default_data_path('train'),
    'val': _get_default_data_path('val'),
    'test': _get_default_data_path('test')
}

# 日志相关常量
DEFAULT_LOG_DIR = 'logs'
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_LOG_INTERVAL = 10

# 分布式训练常量
DEFAULT_MASTER_ADDR = 'localhost'
DEFAULT_MASTER_PORT = '12355'
DEFAULT_BACKEND = 'nccl'  # Linux上使用nccl，Windows上自动切换到gloo

# 模型架构常量
DEFAULT_MODEL_NAME = 'SpectralResNet1D'
DEFAULT_NUM_BLOCKS = 12
DEFAULT_BASE_CHANNELS = 64


class SpectrumConfig:
    """
    光谱配置类，提供统一的配置接口
    """
    
    def __init__(self, spectrum_length: int = SPECTRUM_LENGTH):
        """
        初始化光谱配置
        
        Args:
            spectrum_length: 光谱长度
        """
        if spectrum_length < MIN_SPECTRUM_LENGTH or spectrum_length > MAX_SPECTRUM_LENGTH:
            raise ValueError(
                f"Spectrum length must be between {MIN_SPECTRUM_LENGTH} "
                f"and {MAX_SPECTRUM_LENGTH}, got {spectrum_length}"
            )
        
        self.spectrum_length = spectrum_length
        
    def validate_spectrum(self, spectrum):
        """验证光谱数据"""
        import numpy as np
        
        if len(spectrum) != self.spectrum_length:
            raise ValueError(
                f"Expected spectrum length {self.spectrum_length}, "
                f"got {len(spectrum)}"
            )
        
        if not isinstance(spectrum, (np.ndarray, list)):
            raise TypeError(
                f"Spectrum must be numpy array or list, "
                f"got {type(spectrum)}"
            )
    
    def adjust_spectrum_length(self, spectrum):
        """调整光谱到配置的长度"""
        import numpy as np
        
        current_length = len(spectrum)
        
        if current_length == self.spectrum_length:
            return spectrum
        elif current_length > self.spectrum_length:
            # 截断
            return spectrum[:self.spectrum_length]
        else:
            # 填充
            if isinstance(spectrum, np.ndarray):
                padded = np.zeros(self.spectrum_length, dtype=spectrum.dtype)
                padded[:current_length] = spectrum
                return padded
            else:
                # 列表情况
                return spectrum + [0] * (self.spectrum_length - current_length)


# 全局光谱配置实例
GLOBAL_SPECTRUM_CONFIG = SpectrumConfig(SPECTRUM_LENGTH)


def get_spectrum_length():
    """获取全局光谱长度"""
    return GLOBAL_SPECTRUM_CONFIG.spectrum_length


def set_spectrum_length(length: int):
    """设置全局光谱长度"""
    global GLOBAL_SPECTRUM_CONFIG
    GLOBAL_SPECTRUM_CONFIG = SpectrumConfig(length)


def validate_config_consistency(config: dict):
    """
    验证配置的一致性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 如果配置不一致
    """
    errors = []
    
    # 检查光谱长度一致性
    spectrum_lengths = []
    
    # 收集所有光谱长度相关的配置
    if 'spectrum_length' in config:
        spectrum_lengths.append(('root', config['spectrum_length']))
    
    if 'data' in config and 'spectrum_length' in config['data']:
        spectrum_lengths.append(('data', config['data']['spectrum_length']))
    
    if 'model' in config and 'input_length' in config['model']:
        spectrum_lengths.append(('model', config['model']['input_length']))
    
    if 'metrics' in config and 'spectrum_length' in config['metrics']:
        spectrum_lengths.append(('metrics', config['metrics']['spectrum_length']))
    
    # 检查是否所有长度都一致
    if spectrum_lengths:
        lengths = [length for _, length in spectrum_lengths]
        if len(set(lengths)) > 1:
            error_msg = "Inconsistent spectrum lengths found:\n"
            for location, length in spectrum_lengths:
                error_msg += f"  {location}: {length}\n"
            errors.append(error_msg)
    
    # 检查路径格式一致性（Windows vs Linux）
    paths = []
    if 'train_path' in config:
        paths.append(config['train_path'])
    if 'val_path' in config:
        paths.append(config['val_path'])
    if 'data' in config:
        if 'train_path' in config['data']:
            paths.append(config['data']['train_path'])
        if 'val_path' in config['data']:
            paths.append(config['data']['val_path'])
    
    # 检查路径是否使用了硬编码的绝对路径
    for path in paths:
        path_str = str(path)
        # 检查Windows风格的绝对路径
        if len(path_str) > 2 and path_str[1:3] == ':/':  # C:/, D:/ etc
            errors.append(
                f"Hardcoded Windows absolute path found: {path_str}. "
                "Please use relative paths or environment variables for cross-platform compatibility."
            )
        # 检查Unix风格的绝对路径
        elif path_str.startswith('/home/') or path_str.startswith('/Users/'):
            errors.append(
                f"Hardcoded Unix absolute path found: {path_str}. "
                "Please use relative paths or environment variables for cross-platform compatibility."
            )
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))