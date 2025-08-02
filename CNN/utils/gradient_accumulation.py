import torch
from typing import Optional


class GradientAccumulationManager:
    """
    梯度累积管理器，确保损失缩放的一致性
    """
    def __init__(self, accumulation_steps: int = 1):
        """
        Args:
            accumulation_steps: 梯度累积步数
        """
        self.accumulation_steps = max(1, accumulation_steps)
        self.current_step = 0
        self.accumulated_loss = 0.0
        self.loss_scale = 1.0 / self.accumulation_steps
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失以适应梯度累积
        
        Args:
            loss: 原始损失
            
        Returns:
            缩放后的损失
        """
        return loss * self.loss_scale
    
    def unscale_loss(self, scaled_loss: float) -> float:
        """
        恢复损失的原始尺度（用于日志记录）
        
        Args:
            scaled_loss: 缩放后的损失值
            
        Returns:
            原始尺度的损失值
        """
        return scaled_loss / self.loss_scale
    
    def should_step(self, step: Optional[int] = None) -> bool:
        """
        判断是否应该执行优化器步进
        
        Args:
            step: 当前步数（可选）
            
        Returns:
            是否应该执行优化器步进
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        return (self.current_step % self.accumulation_steps) == 0
    
    def accumulate_loss(self, loss_value: float):
        """
        累积损失值（用于正确的日志记录）
        
        Args:
            loss_value: 当前批次的损失值（已缩放）
        """
        # 恢复原始尺度并累积
        self.accumulated_loss += self.unscale_loss(loss_value)
    
    def get_accumulated_loss(self) -> float:
        """
        获取累积的平均损失
        
        Returns:
            累积步数内的平均损失
        """
        if self.current_step % self.accumulation_steps == 0:
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0  # 重置
            return avg_loss
        return None
    
    def reset(self):
        """重置累积状态"""
        self.current_step = 0
        self.accumulated_loss = 0.0


class GradientAccumulationTrainer:
    """
    支持梯度累积的训练器模板
    """
    def __init__(self, model, optimizer, criterion, accumulation_steps=1, use_amp=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_manager = GradientAccumulationManager(accumulation_steps)
        self.use_amp = use_amp
        
        if use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def train_step(self, inputs, targets):
        """
        执行一个训练步骤
        
        Args:
            inputs: 输入数据
            targets: 目标数据
            
        Returns:
            损失值（原始尺度）
        """
        # 前向传播
        if self.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # 缩放损失
        scaled_loss = self.accumulation_manager.scale_loss(loss)
        
        # 反向传播
        if self.use_amp:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # 记录损失（原始尺度）
        loss_value = loss.item()
        self.accumulation_manager.accumulate_loss(scaled_loss.item())
        
        # 判断是否执行优化器步进
        if self.accumulation_manager.should_step():
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # 返回累积的平均损失
            avg_loss = self.accumulation_manager.get_accumulated_loss()
            return avg_loss
        
        return None  # 还在累积中
    
    def get_effective_batch_size(self, base_batch_size):
        """
        获取有效批次大小
        
        Args:
            base_batch_size: 基础批次大小
            
        Returns:
            考虑梯度累积后的有效批次大小
        """
        return base_batch_size * self.accumulation_manager.accumulation_steps


# 便捷函数
def create_gradient_accumulation_manager(config):
    """
    从配置创建梯度累积管理器
    """
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if 'training' in config:
        accumulation_steps = config['training'].get('gradient_accumulation_steps', accumulation_steps)
    
    return GradientAccumulationManager(accumulation_steps)