import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import os
from .path_utils import normalize_path, get_default_data_path, convert_path_for_config


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 处理环境变量替换
        self._substitute_env_vars(self.config)
        
        return self.config
    
    def _substitute_env_vars(self, config: Dict[str, Any]):
        """递归替换配置中的环境变量"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                default_value = None
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)
                
                config[key] = os.getenv(env_var, default_value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def create_argparser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(description='Spectral Enhancement Training')
        
        # 基础参数
        parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                          help='Path to config file')
        parser.add_argument('--resume', type=str, default=None,
                          help='Path to checkpoint to resume from')
        
        # 训练参数
        parser.add_argument('--batch_size', type=int, default=None,
                          help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=None,
                          help='Learning rate')
        parser.add_argument('--num_epochs', type=int, default=None,
                          help='Number of epochs')
        parser.add_argument('--num_workers', type=int, default=None,
                          help='Number of data loader workers')
        
        # 数据参数
        parser.add_argument('--train_path', type=str, default=None,
                          help='Training data path')
        parser.add_argument('--val_path', type=str, default=None,
                          help='Validation data path')
        parser.add_argument('--data_format', type=str, choices=['h5', 'lmdb', 'mmap'],
                          help='Data format')
        
        # 分布式参数
        parser.add_argument('--gpus', type=int, default=None,
                          help='Number of GPUs to use')
        parser.add_argument('--distributed', action='store_true',
                          help='Enable distributed training')
        
        # 其他参数
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')
        
        return parser
    
    def parse_args_and_update(self, args=None) -> argparse.Namespace:
        """解析命令行参数并更新配置"""
        parser = self.create_argparser()
        parsed_args = parser.parse_args(args)
        
        # 加载配置文件，如果失败则使用默认配置
        if parsed_args.config:
            try:
                self.load_config(parsed_args.config)
            except FileNotFoundError:
                print(f"Warning: Config file {parsed_args.config} not found. Using default configuration.")
                # 加载默认配置
                self._load_default_config()
        
        # 用命令行参数覆盖配置
        updates = {}
        
        if parsed_args.batch_size is not None:
            updates['training.batch_size'] = parsed_args.batch_size
        if parsed_args.learning_rate is not None:
            updates['training.learning_rate'] = parsed_args.learning_rate
        if parsed_args.num_epochs is not None:
            updates['training.num_epochs'] = parsed_args.num_epochs
        if parsed_args.num_workers is not None:
            updates['data.num_workers'] = parsed_args.num_workers
        if parsed_args.train_path is not None:
            updates['data.train_path'] = parsed_args.train_path
        if parsed_args.val_path is not None:
            updates['data.val_path'] = parsed_args.val_path
        if parsed_args.data_format is not None:
            updates['data.format'] = parsed_args.data_format
        if parsed_args.gpus is not None:
            updates['device.num_gpus'] = parsed_args.gpus
        
        # 应用更新
        for key, value in updates.items():
            self.set(key, value)
        
        return parsed_args
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get('data', {})
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.get('loss', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get('logging', {})
    
    def is_distributed(self) -> bool:
        """检查是否启用分布式训练"""
        return self.get('distributed') is not None
    
    def validate_config(self):
        """验证配置的有效性"""
        required_fields = [
            'model.name',
            'training.num_epochs',
            'training.batch_size',
            'data.train_path',
            'data.val_path'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                raise ValueError(f"Required config field missing: {field}")
        
        # 验证路径存在性
        for path_key in ['data.train_path', 'data.val_path']:
            path = self.get(path_key)
            if path and not Path(path).exists():
                print(f"Warning: Path does not exist: {path}")
    
    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("Current Configuration:")
        print("=" * 50)
        self._print_dict(self.config)
        print("=" * 50)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def _load_default_config(self):
        """加载默认配置"""
        self.config = {
            'model': {
                'name': 'SpectralResNet1D',
                'num_blocks': 12,
                'channels': 64,
                'input_channels': 1
            },
            'training': {
                'num_epochs': 100,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'optimizer': 'adam',
                'gradient_clip': 1.0
            },
            'data': {
                'train_path': convert_path_for_config(get_default_data_path('train')),
                'val_path': convert_path_for_config(get_default_data_path('val')),
                'format': 'lmdb',
                'num_workers': 4,
                'cache_size': 100,
                'use_mmap': True,
                'normalize': True
            },
            'loss': {
                'type': 'optimized',
                'peak_weight': 10.0,
                'compton_weight': 1.0,
                'smoothness_weight': 0.1,
                'frequency_weight': 0.1
            },
            'scheduler': {
                'type': 'cosine_annealing_warm_restarts',
                'T_0': 10,
                'T_mult': 2
            },
            'logging': {
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'save_interval': 5,
                'log_interval': 10
            },
            'device': {
                'num_gpus': 1,
                'mixed_precision': False
            },
            'metrics': {
                'use_efficient_metrics': True
            }
        }
        print("Loaded default configuration")


def load_config(config_path: str) -> ConfigManager:
    """便捷函数：加载配置"""
    return ConfigManager(config_path)


def create_default_configs():
    """创建默认配置文件"""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # 默认配置内容
    default_config_content = {
        'model': {
            'name': 'SpectralResNet1D',
            'num_blocks': 12,
            'channels': 64,
            'input_channels': 1
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'gradient_clip': 1.0
        },
        'data': {
            'train_path': convert_path_for_config(get_default_data_path('train')),
            'val_path': convert_path_for_config(get_default_data_path('val')),
            'format': 'lmdb',
            'num_workers': 4,
            'cache_size': 100,
            'use_mmap': True,
            'normalize': True
        },
        'loss': {
            'type': 'optimized',
            'peak_weight': 10.0,
            'compton_weight': 1.0,
            'smoothness_weight': 0.1,
            'frequency_weight': 0.1
        },
        'scheduler': {
            'type': 'cosine_annealing_warm_restarts',
            'T_0': 10,
            'T_mult': 2
        },
        'logging': {
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'save_interval': 5,
            'log_interval': 10
        },
        'device': {
            'num_gpus': 1,
            'mixed_precision': False
        },
        'metrics': {
            'use_efficient_metrics': True
        }
    }
    
    distributed_config_content = default_config_content.copy()
    distributed_config_content.update({
        'training': {
            **default_config_content['training'],
            'batch_size': 128  # 更大的批次大小用于分布式训练
        },
        'data': {
            **default_config_content['data'],
            'num_workers': 16  # 更多的数据加载器工作进程
        },
        'logging': {
            **default_config_content['logging'],
            'log_dir': 'logs_distributed',
            'checkpoint_dir': 'checkpoints_distributed'
        },
        'device': {
            'num_gpus': 4,  # 假设4个GPU
            'mixed_precision': True
        },
        'distributed': {
            'backend': 'nccl',  # Linux上使用nccl，Windows上会自动切换到gloo
            'master_addr': 'localhost',
            'master_port': '12355'
        }
    })
    
    large_scale_config_content = default_config_content.copy()
    large_scale_config_content.update({
        'data': {
            **default_config_content['data'],
            'format': 'lmdb',
            'cache_size': 500,  # 更大的缓存
            'num_workers': 8,
            'use_streaming': True  # 流式加载用于超大数据集
        },
        'training': {
            **default_config_content['training'],
            'batch_size': 8,  # 较小的批次大小以节省内存
            'gradient_accumulation_steps': 4  # 梯度累积
        },
        'model': {
            **default_config_content['model'],
            'name': 'ImprovedSpectralResNet1D',  # 使用改进的模型架构
            'num_blocks': 16,
            'channels': 128
        }
    })
    
    # 配置文件映射
    configs = {
        'default_config.yaml': default_config_content,
        'distributed_config.yaml': distributed_config_content,
        'large_scale_config.yaml': large_scale_config_content
    }
    
    # 创建配置文件
    for config_name, config_content in configs.items():
        config_path = configs_dir / config_name
        if not config_path.exists():
            print(f"Creating default config: {config_path}")
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)