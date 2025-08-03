import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau, ExponentialLR
from typing import Dict, Any, Optional


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器，支持多种类型
    
    Args:
        optimizer: 优化器
        scheduler_config: 调度器配置字典
        
    Returns:
        学习率调度器实例
    """
    scheduler_type = scheduler_config.get('type', 'CosineAnnealingWarmRestarts')
    
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 0)
        )
    
    elif scheduler_type == 'StepLR':
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'ExponentialLR':
        return ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def step_scheduler(scheduler: torch.optim.lr_scheduler._LRScheduler, 
                  epoch: int, 
                  val_loss: Optional[float] = None):
    """
    执行调度器步进，根据调度器类型使用正确的参数
    
    Args:
        scheduler: 学习率调度器
        epoch: 当前epoch
        val_loss: 验证损失（对于ReduceLROnPlateau是必需的）
    """
    if isinstance(scheduler, ReduceLROnPlateau):
        if val_loss is None:
            raise ValueError("ReduceLROnPlateau requires validation loss")
        scheduler.step(val_loss)
    else:
        # 其他调度器不需要参数
        scheduler.step()


def get_scheduler_info(scheduler: torch.optim.lr_scheduler._LRScheduler) -> Dict[str, Any]:
    """
    获取调度器信息用于日志记录
    
    Args:
        scheduler: 学习率调度器
        
    Returns:
        包含调度器信息的字典
    """
    info = {
        'type': scheduler.__class__.__name__,
        'lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None
    }
    
    if isinstance(scheduler, CosineAnnealingWarmRestarts):
        info['T_cur'] = scheduler.T_cur if hasattr(scheduler, 'T_cur') else None
    elif isinstance(scheduler, ReduceLROnPlateau):
        info['num_bad_epochs'] = scheduler.num_bad_epochs if hasattr(scheduler, 'num_bad_epochs') else None
        info['best'] = scheduler.best if hasattr(scheduler, 'best') else None
    
    return info