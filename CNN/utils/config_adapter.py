from typing import Dict, Any, Optional


class ConfigAdapter:
    """
    配置适配器，解决不同训练脚本之间的配置路径不一致问题
    """
    
    @staticmethod
    def get_loss_type(config: Dict[str, Any]) -> str:
        """
        获取损失类型，兼容不同的配置路径
        
        Args:
            config: 配置字典
            
        Returns:
            损失类型字符串
        """
        # 尝试不同的路径
        loss_type = None
        
        # 路径1: 直接在根级别
        if 'loss_type' in config:
            loss_type = config['loss_type']
        
        # 路径2: 在loss配置下
        elif 'loss' in config and isinstance(config['loss'], dict):
            if 'type' in config['loss']:
                loss_type = config['loss']['type']
            elif 'loss_type' in config['loss']:
                loss_type = config['loss']['loss_type']
        
        # 路径3: 在training配置下
        elif 'training' in config and isinstance(config['training'], dict):
            if 'loss_type' in config['training']:
                loss_type = config['training']['loss_type']
        
        # 默认值
        if loss_type is None:
            loss_type = 'original'
        
        return loss_type
    
    @staticmethod
    def get_loss_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取完整的损失函数配置，统一不同的配置格式
        
        Args:
            config: 配置字典
            
        Returns:
            统一格式的损失函数配置
        """
        loss_config = {
            'type': ConfigAdapter.get_loss_type(config),
            'peak_weight': 10.0,
            'compton_weight': 1.0,
            'smoothness_weight': 0.1,
            'frequency_weight': 0.1
        }
        
        # 从不同位置收集损失配置参数
        
        # 源1: 根级别参数
        if 'peak_weight' in config:
            loss_config['peak_weight'] = config['peak_weight']
        if 'compton_weight' in config:
            loss_config['compton_weight'] = config['compton_weight']
        if 'base_weight' in config:  # compton_weight的别名
            loss_config['compton_weight'] = config['base_weight']
        if 'smoothness_weight' in config:
            loss_config['smoothness_weight'] = config['smoothness_weight']
        if 'frequency_weight' in config:
            loss_config['frequency_weight'] = config['frequency_weight']
        
        # 源2: loss配置部分
        if 'loss' in config and isinstance(config['loss'], dict):
            loss_dict = config['loss']
            if 'peak_weight' in loss_dict:
                loss_config['peak_weight'] = loss_dict['peak_weight']
            if 'compton_weight' in loss_dict:
                loss_config['compton_weight'] = loss_dict['compton_weight']
            if 'base_weight' in loss_dict:
                loss_config['compton_weight'] = loss_dict['base_weight']
            if 'smoothness_weight' in loss_dict:
                loss_config['smoothness_weight'] = loss_dict['smoothness_weight']
            if 'frequency_weight' in loss_dict:
                loss_config['frequency_weight'] = loss_dict['frequency_weight']
        
        return loss_config
    
    @staticmethod
    def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取训练配置，统一格式
        """
        training_config = {
            'num_epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'optimizer': 'adam'
        }
        
        # 从根级别获取
        for key in training_config:
            if key in config:
                training_config[key] = config[key]
        
        # 从training部分获取
        if 'training' in config and isinstance(config['training'], dict):
            for key in training_config:
                if key in config['training']:
                    training_config[key] = config['training'][key]
        
        return training_config
    
    @staticmethod
    def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取数据配置，统一格式
        """
        data_config = {
            'train_path': '',
            'val_path': '',
            'num_workers': 4,
            'cache_size': 100,
            'use_mmap': True,
            'normalize': True,
            'format': 'h5',
            'use_isolated_cache': True
        }
        
        # 从根级别获取
        for key in ['train_path', 'val_path', 'num_workers', 'cache_size']:
            if key in config:
                data_config[key] = config[key]
        
        # 从data部分获取
        if 'data' in config and isinstance(config['data'], dict):
            for key in data_config:
                if key in config['data']:
                    data_config[key] = config['data'][key]
        
        return data_config
    
    @staticmethod
    def adapt_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将任意格式的配置适配为统一格式
        
        Args:
            config: 原始配置
            
        Returns:
            统一格式的配置
        """
        adapted = {
            'loss': ConfigAdapter.get_loss_config(config),
            'training': ConfigAdapter.get_training_config(config),
            'data': ConfigAdapter.get_data_config(config),
            'device': config.get('device', {}),
            'logging': config.get('logging', {
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'save_interval': 5
            }),
            'metrics': config.get('metrics', {
                'use_efficient_metrics': True
            })
        }
        
        # 保留其他未知配置
        for key in config:
            if key not in ['loss', 'training', 'data', 'device', 'logging', 'metrics']:
                # 如果不是已知的嵌套配置，保留在根级别
                if key not in [
                    'loss_type', 'peak_weight', 'compton_weight', 'base_weight',
                    'smoothness_weight', 'frequency_weight', 'num_epochs', 'batch_size',
                    'learning_rate', 'weight_decay', 'gradient_clip', 'optimizer',
                    'train_path', 'val_path', 'num_workers', 'cache_size'
                ]:
                    adapted[key] = config[key]
        
        return adapted
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个配置，override_config优先级更高
        """
        import copy
        merged = copy.deepcopy(base_config)
        
        def deep_update(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(merged, override_config)
        return merged


# 便捷函数
def get_unified_loss_config(config):
    """获取统一的损失函数配置"""
    return ConfigAdapter.get_loss_config(config)


def adapt_legacy_config(config):
    """适配旧版配置格式"""
    return ConfigAdapter.adapt_config(config)