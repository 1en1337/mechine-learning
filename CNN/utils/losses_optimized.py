import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


class OptimizedPeakAwareLoss(nn.Module):
    """
    优化的峰感知损失函数，解决性能瓶颈和低效逻辑
    """
    def __init__(self, peak_weight=10.0, base_weight=1.0, 
                 smoothness_weight=0.1, frequency_weight=0.1):
        super().__init__()
        self.peak_weight = peak_weight
        self.base_weight = base_weight
        self.smoothness_weight = smoothness_weight
        self.frequency_weight = frequency_weight
        
        # 预创建和缓存常用的核
        self._register_kernels()
        
        # 缓存窗函数
        self._cached_windows = {}
        
    def _register_kernels(self):
        """注册常用的卷积核为buffer，避免重复创建"""
        # 峰检测核
        kernel_size = 21
        x = torch.linspace(-3, 3, kernel_size, dtype=torch.float32)
        x_squared = x * x
        exp_term = torch.exp(-0.5 * x_squared)
        normalization = 2.0 / (1.7320508 * 1.3313353)
        mexican_hat = normalization * (1 - x_squared) * exp_term
        max_val = mexican_hat.abs().max()
        if max_val > 0:
            mexican_hat = mexican_hat / max_val
        
        self.register_buffer('peak_kernel', mexican_hat.view(1, 1, -1))
        
        # 平滑核（用于计算平滑性）
        smooth_kernel = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer('smooth_kernel', smooth_kernel)
    
    def _get_hann_window(self, size, device):
        """获取缓存的Hann窗，避免重复创建"""
        if size not in self._cached_windows:
            self._cached_windows[size] = torch.hann_window(size, device=device)
        return self._cached_windows[size]
    
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, pred, target):
        # 确保输入形状一致
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)
        
        batch_size, channels, length = pred.shape
        device = pred.device
        
        # 批量计算MSE损失
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # 优化的峰检测 - 批量处理所有通道
        # 使用预先注册的kernel避免重复创建
        peak_kernel = self.peak_kernel.to(device)
        
        # 合并批次和通道维度进行单次卷积
        pred_flat = pred.view(-1, 1, length)
        target_flat = target.view(-1, 1, length)
        
        # 批量峰检测
        pred_peaks = F.conv1d(pred_flat, peak_kernel, padding=10)
        target_peaks = F.conv1d(target_flat, peak_kernel, padding=10)
        
        # 恢复原始形状
        pred_peaks = pred_peaks.view(batch_size, channels, length)
        target_peaks = target_peaks.view(batch_size, channels, length)
        
        # 峰区域重要性 - 使用更高效的操作
        peak_importance = torch.maximum(
            torch.abs(target_peaks),
            torch.abs(pred_peaks) * 0.5
        )
        
        # 归一化峰重要性
        peak_importance = peak_importance / (peak_importance.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # 使用软掩码而不是硬掩码，避免梯度截断
        peak_weight_map = torch.sigmoid((peak_importance - 0.5) * 10)  # 软阈值
        base_weight_map = 1 - peak_weight_map
        
        # 加权损失
        weighted_loss = mse_loss * (self.peak_weight * peak_weight_map + 
                                   self.base_weight * base_weight_map)
        base_loss = weighted_loss.mean()
        
        # 优化的平滑性损失 - 使用卷积而不是切片
        if self.smoothness_weight > 0:
            smooth_kernel = self.smooth_kernel.to(device)
            pred_smoothness = F.conv1d(pred_flat, smooth_kernel, padding=1)
            target_smoothness = F.conv1d(target_flat, smooth_kernel, padding=1)
            smoothness_loss = F.mse_loss(pred_smoothness, target_smoothness)
        else:
            smoothness_loss = 0.0
        
        # 优化的频域损失
        if self.frequency_weight > 0:
            # 使用缓存的窗函数
            window = self._get_hann_window(length, device)
            
            # 批量FFT
            windowed_pred = pred * window
            windowed_target = target * window
            
            # 使用2D FFT更高效地处理批量数据
            pred_fft = torch.fft.rfft(windowed_pred, dim=-1)
            target_fft = torch.fft.rfft(windowed_target, dim=-1)
            
            # 对数幅度谱
            pred_mag = torch.log1p(torch.abs(pred_fft))
            target_mag = torch.log1p(torch.abs(target_fft))
            
            frequency_loss = F.mse_loss(pred_mag, target_mag)
        else:
            frequency_loss = 0.0
        
        # 峰对齐损失 - 使用相关性而不是MSE
        peak_alignment = F.cosine_similarity(
            pred_peaks.flatten(1), 
            target_peaks.flatten(1), 
            dim=1
        ).mean()
        peak_alignment_loss = 1 - peak_alignment  # 转换为损失
        
        # 总损失
        total_loss = (base_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.frequency_weight * frequency_loss +
                     0.1 * peak_alignment_loss)
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/Inf detected, using fallback loss")
            total_loss = F.mse_loss(pred, target)
        
        # 返回损失和组件
        return total_loss, {
            'base_loss': base_loss.item(),
            'smoothness_loss': smoothness_loss.item() if torch.is_tensor(smoothness_loss) else smoothness_loss,
            'frequency_loss': frequency_loss.item() if torch.is_tensor(frequency_loss) else frequency_loss,
            'peak_alignment_loss': peak_alignment_loss.item()
        }


class BatchOptimizedLoss(nn.Module):
    """
    批量优化的损失函数，充分利用GPU并行计算
    """
    def __init__(self, loss_config):
        super().__init__()
        self.loss_config = loss_config
        
        # 创建多个损失函数组件
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # 权重
        self.weights = {
            'mse': loss_config.get('mse_weight', 1.0),
            'l1': loss_config.get('l1_weight', 0.1),
            'gradient': loss_config.get('gradient_weight', 0.1),
            'frequency': loss_config.get('frequency_weight', 0.1)
        }
        
    def forward(self, pred_batch, target_batch):
        """
        批量处理多个损失组件
        Args:
            pred_batch: [B, C, L] 预测张量
            target_batch: [B, C, L] 目标张量
        """
        device = pred_batch.device
        total_loss = 0
        loss_dict = {}
        
        # 1. 基础损失 - 批量计算
        if self.weights['mse'] > 0:
            mse = self.mse_loss(pred_batch, target_batch).mean()
            total_loss = total_loss + self.weights['mse'] * mse
            loss_dict['mse'] = mse.item()
        
        if self.weights['l1'] > 0:
            l1 = self.l1_loss(pred_batch, target_batch).mean()
            total_loss = total_loss + self.weights['l1'] * l1
            loss_dict['l1'] = l1.item()
        
        # 2. 梯度损失 - 使用卷积批量计算
        if self.weights['gradient'] > 0:
            # 创建梯度核
            grad_kernel = torch.tensor([-1.0, 0.0, 1.0], device=device).view(1, 1, -1)
            
            # 展平批次维度
            B, C, L = pred_batch.shape
            pred_flat = pred_batch.view(B * C, 1, L)
            target_flat = target_batch.view(B * C, 1, L)
            
            # 批量梯度计算
            pred_grad = F.conv1d(pred_flat, grad_kernel, padding=1)
            target_grad = F.conv1d(target_flat, grad_kernel, padding=1)
            
            grad_loss = F.mse_loss(pred_grad, target_grad)
            total_loss = total_loss + self.weights['gradient'] * grad_loss
            loss_dict['gradient'] = grad_loss.item()
        
        # 3. 频域损失 - 批量FFT
        if self.weights['frequency'] > 0:
            # 批量FFT（自动处理所有维度）
            pred_fft = torch.fft.rfft(pred_batch, dim=-1)
            target_fft = torch.fft.rfft(target_batch, dim=-1)
            
            # 幅度和相位
            pred_mag = torch.log1p(torch.abs(pred_fft))
            target_mag = torch.log1p(torch.abs(target_fft))
            
            freq_loss = F.mse_loss(pred_mag, target_mag)
            total_loss = total_loss + self.weights['frequency'] * freq_loss
            loss_dict['frequency'] = freq_loss.item()
        
        return total_loss, loss_dict


class EfficientCompositeLoss(nn.Module):
    """
    高效的复合损失函数，整合多种损失类型
    """
    def __init__(self, config):
        super().__init__()
        
        # 主损失
        self.peak_aware_loss = OptimizedPeakAwareLoss(
            peak_weight=config.get('peak_weight', 10.0),
            base_weight=config.get('base_weight', 1.0),
            smoothness_weight=config.get('smoothness_weight', 0.1),
            frequency_weight=config.get('frequency_weight', 0.1)
        )
        
        # 辅助损失
        self.use_auxiliary = config.get('use_auxiliary_losses', False)
        if self.use_auxiliary:
            self.auxiliary_loss = BatchOptimizedLoss(config.get('auxiliary_config', {}))
            self.auxiliary_weight = config.get('auxiliary_weight', 0.1)
    
    def forward(self, pred, target):
        # 主损失
        main_loss, main_components = self.peak_aware_loss(pred, target)
        
        total_loss = main_loss
        all_components = main_components.copy()
        
        # 辅助损失
        if self.use_auxiliary:
            aux_loss, aux_components = self.auxiliary_loss(pred, target)
            total_loss = total_loss + self.auxiliary_weight * aux_loss
            
            # 添加辅助损失组件
            for key, value in aux_components.items():
                all_components[f'aux_{key}'] = value
        
        return total_loss, all_components