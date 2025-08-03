from typing import Dict, Tuple
import torch


class LossComponentAdapter:
    """
    适配器类，统一不同损失函数返回的组件键名
    确保所有损失函数返回一致的组件结构
    """
    
    # 标准组件键
    STANDARD_KEYS = {
        'peak_loss',
        'compton_loss',
        'smoothness_loss',
        'total_loss'
    }
    
    # 键映射关系
    KEY_MAPPINGS = {
        'non_peak_loss': 'compton_loss',  # OptimizedSpectralLoss使用non_peak_loss
        'base_loss': 'compton_loss',      # 某些损失函数可能使用base_loss
    }
    
    @staticmethod
    def adapt_components(components: Dict[str, float]) -> Dict[str, float]:
        """
        适配损失组件，确保返回标准键
        
        Args:
            components: 原始损失组件字典
            
        Returns:
            标准化的损失组件字典
        """
        adapted = {}
        
        # 首先复制所有原始键
        for key, value in components.items():
            adapted[key] = value
        
        # 应用键映射
        for old_key, new_key in LossComponentAdapter.KEY_MAPPINGS.items():
            if old_key in components and new_key not in adapted:
                adapted[new_key] = components[old_key]
        
        # 确保所有标准键都存在
        for key in LossComponentAdapter.STANDARD_KEYS:
            if key not in adapted:
                # 设置默认值
                if key == 'total_loss':
                    # 如果没有total_loss，尝试计算
                    total = 0.0
                    if 'peak_loss' in adapted:
                        total += adapted['peak_loss']
                    if 'compton_loss' in adapted:
                        total += adapted['compton_loss']
                    if 'smoothness_loss' in adapted:
                        total += adapted['smoothness_loss']
                    adapted[key] = total if total > 0 else 0.0
                else:
                    adapted[key] = 0.0
        
        return adapted
    
    @staticmethod
    def create_loss_wrapper(loss_fn):
        """
        创建损失函数包装器，自动适配组件
        
        Args:
            loss_fn: 原始损失函数
            
        Returns:
            包装后的损失函数
        """
        class WrappedLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_fn = loss_fn
            
            def forward(self, pred, target):
                # 调用原始损失函数
                loss, components = self.loss_fn(pred, target)
                
                # 适配组件
                adapted_components = LossComponentAdapter.adapt_components(components)
                
                return loss, adapted_components
        
        return WrappedLoss()
    
    @staticmethod
    def validate_components(components: Dict[str, float]) -> bool:
        """
        验证损失组件是否包含所有必需的键
        
        Args:
            components: 损失组件字典
            
        Returns:
            是否有效
        """
        return all(key in components for key in LossComponentAdapter.STANDARD_KEYS)
    
    @staticmethod
    def merge_components(components_list: list) -> Dict[str, float]:
        """
        合并多个损失组件字典（用于分布式训练）
        
        Args:
            components_list: 损失组件字典列表
            
        Returns:
            合并后的损失组件字典
        """
        if not components_list:
            return {key: 0.0 for key in LossComponentAdapter.STANDARD_KEYS}
        
        merged = {}
        
        # 获取所有唯一的键
        all_keys = set()
        for components in components_list:
            all_keys.update(components.keys())
        
        # 对每个键求平均
        for key in all_keys:
            values = [c.get(key, 0.0) for c in components_list if key in c]
            if values:
                merged[key] = sum(values) / len(values)
        
        # 确保标准键存在
        return LossComponentAdapter.adapt_components(merged)