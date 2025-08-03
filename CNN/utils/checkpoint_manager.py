import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import json
from torch.cuda.amp import GradScaler


class CheckpointManager:
    """
    安全的检查点管理器，处理各种兼容性问题
    """
    
    # 当前检查点版本
    CHECKPOINT_VERSION = "1.0"
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       val_loss: float,
                       config: Dict[str, Any],
                       scaler: Optional[GradScaler] = None,
                       is_best: bool = False,
                       extra_data: Optional[Dict[str, Any]] = None) -> str:
        """
        保存检查点，包含完整的训练状态和元信息
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            val_loss: 验证损失
            config: 配置信息
            scaler: AMP缩放器
            is_best: 是否是最佳模型
            extra_data: 额外数据
            
        Returns:
            保存的检查点路径
        """
        # 获取模型状态（处理DDP）
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
            model_name = model.module.__class__.__name__
        else:
            model_state_dict = model.state_dict()
            model_name = model.__class__.__name__
        
        # 构建检查点数据
        checkpoint_data = {
            # 版本信息
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'pytorch_version': torch.__version__,
            
            # 模型信息
            'model_name': model_name,
            'model_state_dict': model_state_dict,
            'model_config': self._extract_model_config(config),
            
            # 训练状态
            'epoch': epoch,
            'val_loss': val_loss,
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_name': optimizer.__class__.__name__,
            
            # 调度器状态
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scheduler_name': scheduler.__class__.__name__ if scheduler else None,
            
            # AMP状态
            'use_amp': scaler is not None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            
            # 配置信息
            'config': config,
            'training_config': self._extract_training_config(config),
            
            # 额外数据
            'extra_data': extra_data or {}
        }
        
        # 保存路径
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        
        # 保存模型配置信息（便于调试）
        if is_best:
            config_path = self.checkpoint_dir / 'best_model_config.json'
            self._save_readable_config(config, config_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self,
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       scaler: Optional[GradScaler] = None,
                       strict_loading: bool = False,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        安全地加载检查点，处理各种兼容性问题
        
        Args:
            checkpoint_path: 检查点路径
            model: 目标模型
            optimizer: 目标优化器
            scheduler: 目标调度器
            scaler: 目标AMP缩放器
            strict_loading: 是否严格加载（失败时抛出异常）
            device: 目标设备
            
        Returns:
            包含恢复信息的字典
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # 加载检查点数据
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
        except Exception as e:
            if strict_loading:
                raise e
            else:
                print(f"Warning: Failed to load checkpoint: {e}")
                return self._create_default_resume_info()
        
        # 验证检查点
        validation_result = self._validate_checkpoint(checkpoint, model, optimizer, scheduler, scaler)
        
        if not validation_result['valid'] and strict_loading:
            raise ValueError(f"Checkpoint validation failed: {validation_result['errors']}")
        
        # 加载模型状态
        model_loaded = self._load_model_state(checkpoint, model, strict_loading)
        
        # 加载优化器状态
        optimizer_loaded = self._load_optimizer_state(checkpoint, optimizer, strict_loading)
        
        # 加载调度器状态
        scheduler_loaded = self._load_scheduler_state(checkpoint, scheduler, strict_loading)
        
        # 加载AMP状态
        scaler_loaded = self._load_scaler_state(checkpoint, scaler, strict_loading)
        
        # 构建恢复信息
        resume_info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'start_epoch': checkpoint.get('epoch', 0) + 1,
            'model_loaded': model_loaded,
            'optimizer_loaded': optimizer_loaded,
            'scheduler_loaded': scheduler_loaded,
            'scaler_loaded': scaler_loaded,
            'validation_result': validation_result,
            'checkpoint_config': checkpoint.get('config', {}),
            'extra_data': checkpoint.get('extra_data', {})
        }
        
        print(f"Checkpoint loaded successfully. Resuming from epoch {resume_info['start_epoch']}")
        
        return resume_info
    
    def _validate_checkpoint(self, checkpoint: Dict, model: nn.Module, optimizer: torch.optim.Optimizer,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                            scaler: Optional[GradScaler]) -> Dict[str, Any]:
        """验证检查点兼容性"""
        errors = []
        warnings_list = []
        
        # 检查版本
        checkpoint_version = checkpoint.get('checkpoint_version', 'unknown')
        if checkpoint_version != self.CHECKPOINT_VERSION:
            warnings_list.append(f"Checkpoint version mismatch: {checkpoint_version} vs {self.CHECKPOINT_VERSION}")
        
        # 检查模型
        current_model_name = model.module.__class__.__name__ if hasattr(model, 'module') else model.__class__.__name__
        checkpoint_model_name = checkpoint.get('model_name', 'unknown')
        if current_model_name != checkpoint_model_name:
            warnings_list.append(f"Model type mismatch: {current_model_name} vs {checkpoint_model_name}")
        
        # 检查优化器
        current_optimizer_name = optimizer.__class__.__name__
        checkpoint_optimizer_name = checkpoint.get('optimizer_name', 'unknown')
        if current_optimizer_name != checkpoint_optimizer_name:
            warnings_list.append(f"Optimizer type mismatch: {current_optimizer_name} vs {checkpoint_optimizer_name}")
        
        # 检查调度器
        if scheduler:
            current_scheduler_name = scheduler.__class__.__name__
            checkpoint_scheduler_name = checkpoint.get('scheduler_name', None)
            if checkpoint_scheduler_name and current_scheduler_name != checkpoint_scheduler_name:
                warnings_list.append(f"Scheduler type mismatch: {current_scheduler_name} vs {checkpoint_scheduler_name}")
        
        # 检查AMP状态
        current_use_amp = scaler is not None
        checkpoint_use_amp = checkpoint.get('use_amp', False)
        if current_use_amp != checkpoint_use_amp:
            warnings_list.append(f"AMP usage mismatch: current={current_use_amp}, checkpoint={checkpoint_use_amp}")
        
        # 检查必需的键
        required_keys = ['model_state_dict', 'epoch']
        for key in required_keys:
            if key not in checkpoint:
                errors.append(f"Missing required key: {key}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings_list
        }
    
    def _load_model_state(self, checkpoint: Dict, model: nn.Module, strict: bool) -> bool:
        """安全加载模型状态"""
        try:
            model_state_dict = checkpoint['model_state_dict']
            
            # 处理DDP键名不匹配
            current_keys = set(model.state_dict().keys())
            checkpoint_keys = set(model_state_dict.keys())
            
            # 检查键名前缀不匹配（DDP vs non-DDP）
            if not current_keys.intersection(checkpoint_keys):
                # 尝试添加/移除 'module.' 前缀
                new_state_dict = {}
                if any(k.startswith('module.') for k in checkpoint_keys):
                    # 检查点有module前缀，当前模型没有
                    for k, v in model_state_dict.items():
                        new_key = k[7:] if k.startswith('module.') else k
                        new_state_dict[new_key] = v
                else:
                    # 检查点没有module前缀，当前模型有
                    for k, v in model_state_dict.items():
                        new_key = f'module.{k}' if not k.startswith('module.') else k
                        new_state_dict[new_key] = v
                model_state_dict = new_state_dict
            
            # 加载状态字典
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in model: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            if strict and (missing_keys or unexpected_keys):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading model state: {e}")
            if strict:
                raise e
            return False
    
    def _load_optimizer_state(self, checkpoint: Dict, optimizer: torch.optim.Optimizer, strict: bool) -> bool:
        """安全加载优化器状态"""
        try:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return True
            else:
                print("Warning: No optimizer state in checkpoint")
                return False
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")
            if strict:
                raise e
            return False
    
    def _load_scheduler_state(self, checkpoint: Dict, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], strict: bool) -> bool:
        """安全加载调度器状态"""
        if not scheduler:
            return True
        
        try:
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                return True
            else:
                print("Warning: No scheduler state in checkpoint")
                return False
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}")
            if strict:
                raise e
            return False
    
    def _load_scaler_state(self, checkpoint: Dict, scaler: Optional[GradScaler], strict: bool) -> bool:
        """安全加载AMP缩放器状态"""
        checkpoint_has_amp = checkpoint.get('use_amp', False)
        current_has_amp = scaler is not None
        
        if checkpoint_has_amp and current_has_amp:
            try:
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    return True
            except Exception as e:
                print(f"Warning: Failed to load scaler state: {e}")
                if strict:
                    raise e
        elif checkpoint_has_amp != current_has_amp:
            print(f"Warning: AMP usage mismatch - checkpoint: {checkpoint_has_amp}, current: {current_has_amp}")
        
        return True
    
    def _extract_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取模型相关配置"""
        return config.get('model', {})
    
    def _extract_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """提取训练相关配置"""
        return {
            'training': config.get('training', {}),
            'loss': config.get('loss', {}),
            'scheduler': config.get('scheduler', {})
        }
    
    def _save_readable_config(self, config: Dict[str, Any], config_path: Path):
        """保存可读的配置文件"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save config file: {e}")
    
    def _create_default_resume_info(self) -> Dict[str, Any]:
        """创建默认的恢复信息"""
        return {
            'epoch': 0,
            'val_loss': float('inf'),
            'start_epoch': 0,
            'model_loaded': False,
            'optimizer_loaded': False,
            'scheduler_loaded': False,
            'scaler_loaded': False,
            'validation_result': {'valid': False, 'errors': ['Failed to load checkpoint'], 'warnings': []},
            'checkpoint_config': {},
            'extra_data': {}
        }
    
    def list_checkpoints(self) -> list:
        """列出所有可用的检查点"""
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob('*.pth'):
            checkpoints.append(str(checkpoint_file))
        return sorted(checkpoints)
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """清理旧的检查点，保留最新的几个"""
        epoch_checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob('checkpoint_epoch_*.pth'):
            try:
                epoch_num = int(checkpoint_file.stem.split('_')[-1])
                epoch_checkpoints.append((epoch_num, checkpoint_file))
            except ValueError:
                continue
        
        # 按epoch排序，删除旧的
        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
        for epoch_num, checkpoint_file in epoch_checkpoints[keep_last:]:
            try:
                checkpoint_file.unlink()
                print(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"Warning: Failed to remove {checkpoint_file}: {e}")


# 便捷函数
def create_checkpoint_manager(config: Dict[str, Any]) -> CheckpointManager:
    """从配置创建检查点管理器"""
    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'checkpoints')
    return CheckpointManager(checkpoint_dir)