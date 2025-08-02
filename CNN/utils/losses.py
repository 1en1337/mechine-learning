import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks
from .loss_adapter import LossComponentAdapter


class OptimizedSpectralLoss(nn.Module):
    """优化的光谱损失函数 - 纯GPU实现，高性能"""
    
    def __init__(self, peak_weight=10.0, base_weight=1.0, smoothness_weight=0.1, 
                 frequency_weight=0.1, peak_width=25, min_peak_height=0.1):
        super(OptimizedSpectralLoss, self).__init__()
        self.peak_weight = peak_weight
        self.base_weight = base_weight
        self.smoothness_weight = smoothness_weight
        self.frequency_weight = frequency_weight
        self.peak_width = peak_width
        self.min_peak_height = min_peak_height
        
        # 创建峰检测卷积核（墨西哥帽小波）
        self.register_buffer('peak_detector_kernel', self._create_peak_detector_kernel())
        
    def _create_peak_detector_kernel(self, kernel_size=21):
        """创建墨西哥帽小波核用于峰检测 - 改进数值稳定性"""
        # 使用更稳定的计算方式
        x = torch.linspace(-3, 3, kernel_size, dtype=torch.float32)
        # 避免灾难性取消，分步计算
        x_squared = x * x
        exp_term = torch.exp(-0.5 * x_squared)
        
        # 使用更稳定的常数
        normalization = 2.0 / (1.7320508 * 1.3313353)  # sqrt(3) * pi^0.25
        
        mexican_hat = normalization * (1 - x_squared) * exp_term
        
        # 更稳健的归一化
        max_val = mexican_hat.abs().max()
        if max_val > 0:
            mexican_hat = mexican_hat / max_val
        
        return mexican_hat.view(1, 1, -1)
    
    def gpu_peak_detection(self, spectrum):
        """GPU原生峰检测，避免CPU转换"""
        # 确保输入是3维张量 [batch, channel, length]
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
        batch_size, channels, length = spectrum.shape
        
        # 使用墨西哥帽小波进行峰检测 - 确保设备匹配
        # 将kernel移动到与输入相同的设备
        kernel = self.peak_detector_kernel.to(spectrum.device)
        padding = kernel.size(-1) // 2
        peak_response = F.conv1d(spectrum, kernel, padding=padding)
        
        # 找到局部最大值
        max_pool = F.max_pool1d(peak_response, kernel_size=21, stride=1, padding=10)
        local_maxima = (peak_response == max_pool) & (peak_response > 0)
        
        # 动态阈值：基于每个样本的最大值
        max_values = spectrum.max(dim=-1, keepdim=True)[0]
        threshold = max_values * self.min_peak_height
        significant_peaks = local_maxima & (spectrum > threshold)
        
        # 扩展峰区域
        peak_regions = self._expand_peak_regions(significant_peaks)
        
        return peak_regions.float()
    
    def _expand_peak_regions(self, peak_points):
        """扩展峰点为峰区域"""
        batch_size, channels, length = peak_points.shape
        device = peak_points.device
        
        # 创建扩展核
        expand_kernel = torch.ones(1, 1, self.peak_width * 2 + 1, device=device)
        padding = self.peak_width
        
        # 使用卷积扩展峰区域
        peak_regions = F.conv1d(peak_points.float(), expand_kernel, padding=padding)
        peak_regions = (peak_regions > 0).float()
        
        return peak_regions
    
    def forward(self, pred, target):
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # GPU峰检测 - 基于目标和预测的峰区域
        target_peak_regions = self.gpu_peak_detection(target)
        pred_peak_regions = self.gpu_peak_detection(pred)
        
        # 公平的峰区域组合 - 同时考虑目标和预测的峰
        # 这样可以鼓励模型发现新的峰，而不仅仅是复制目标峰
        peak_importance = torch.maximum(target_peak_regions, pred_peak_regions * 0.5)
        
        # 峰区域和非峰区域的损失 - 添加NaN检查
        peak_mask = peak_importance > 0.5
        
        # 避免除零和NaN传播
        peak_count = peak_mask.sum()
        non_peak_count = (~peak_mask).sum()
        
        if peak_count > 0:
            peak_loss = (mse_loss * peak_mask).sum() / peak_count
        else:
            peak_loss = torch.tensor(0.0, device=mse_loss.device)
            
        if non_peak_count > 0:
            non_peak_loss = (mse_loss * (~peak_mask)).sum() / non_peak_count
        else:
            non_peak_loss = torch.tensor(0.0, device=mse_loss.device)
        
        # 检查NaN
        if torch.isnan(peak_loss) or torch.isnan(non_peak_loss):
            raise ValueError("NaN detected in loss calculation")
        
        # 自适应权重损失
        base_loss = self.peak_weight * peak_loss + self.base_weight * non_peak_loss
        
        # 平滑性损失 - 比较预测和目标的平滑性差异
        pred_smooth = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        target_smooth = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        smoothness_loss = F.mse_loss(pred_smooth, target_smooth, reduction='mean')
        
        # 峰对齐损失 - 使用加权的方式确保重要峰的对齐
        # 这避免了对所有位置给予相同权重
        peak_weights = target_peak_regions + 0.1  # 加小值避免完全忽略非峰区域
        peak_alignment_loss = (F.mse_loss(pred_peak_regions, target_peak_regions, reduction='none') * peak_weights).mean()
        
        # 频域损失（可选）
        # 频域损失 - 添加窗函数避免频谱泄漏
        frequency_loss = 0.0
        if self.frequency_weight > 0:
            # 应用Hann窗减少频谱泄漏
            window = torch.hann_window(pred.size(-1), device=pred.device)
            windowed_pred = pred * window
            windowed_target = target * window
            
            pred_fft = torch.fft.rfft(windowed_pred, dim=-1)
            target_fft = torch.fft.rfft(windowed_target, dim=-1)
            
            # 使用对数刻度避免数值问题
            pred_mag = torch.log1p(torch.abs(pred_fft))  # log(1 + x) 避免 log(0)
            target_mag = torch.log1p(torch.abs(target_fft))
            
            frequency_loss = F.mse_loss(pred_mag, target_mag, reduction='mean')
        
        # 总损失 - 添加NaN检查和剪裁
        total_loss = (base_loss + 
                     self.smoothness_weight * smoothness_loss + 
                     self.frequency_weight * frequency_loss +
                     0.1 * peak_alignment_loss)  # 峰对齐权重
        
        # 检查和处理NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected in loss. base_loss={base_loss}, "
                  f"smoothness_loss={smoothness_loss}, frequency_loss={frequency_loss}")
            # 返回一个安全的默认损失
            total_loss = F.mse_loss(pred, target, reduction='mean')
        
        # 剪裁极端值
        total_loss = torch.clamp(total_loss, min=0.0, max=1e6)
        
        components = {
            'peak_loss': peak_loss.item(),
            'non_peak_loss': non_peak_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'peak_alignment_loss': peak_alignment_loss.item(),
            'frequency_loss': frequency_loss.item() if isinstance(frequency_loss, torch.Tensor) else frequency_loss,
            'total_loss': total_loss.item()
        }
        
        # 使用适配器确保返回标准键
        adapted_components = LossComponentAdapter.adapt_components(components)
        # 保留额外的键以便调试
        adapted_components.update({
            'peak_alignment_loss': components['peak_alignment_loss'],
            'frequency_loss': components['frequency_loss']
        })
        
        return total_loss, adapted_components


class SpectralCompositeLoss(nn.Module):
    """保留原有接口的兼容版本，内部使用优化实现"""
    
    def __init__(self, peak_weight=10.0, compton_weight=1.0, smoothness_weight=0.1):
        super(SpectralCompositeLoss, self).__init__()
        # 使用优化版本，但保持相同的参数接口
        self.optimized_loss = OptimizedSpectralLoss(
            peak_weight=peak_weight,
            base_weight=compton_weight,
            smoothness_weight=smoothness_weight
        )
        
    def forward(self, pred, target):
        total_loss, components = self.optimized_loss(pred, target)
        
        # 为了兼容性，返回所需的字段
        return total_loss, {
            'peak_loss': components.get('peak_loss', 0.0),
            'compton_loss': components.get('non_peak_loss', 0.0)  # 使用non_peak_loss作为compton_loss
        }
    
    # 保留原有方法以防某些代码调用
    def find_peak_regions(self, spectrum):
        """兼容性方法，使用GPU优化版本"""
        return self.optimized_loss.gpu_peak_detection(spectrum)


class AdaptivePeakWeightedLoss(nn.Module):
    """可学习的自适应峰权重损失函数"""
    
    def __init__(self, base_weight=1.0, max_peak_weight=10.0, smoothness_weight=0.1):
        super(AdaptivePeakWeightedLoss, self).__init__()
        self.base_weight = base_weight
        self.max_peak_weight = max_peak_weight
        self.smoothness_weight = smoothness_weight
        
        # 可学习的峰检测网络
        self.peak_detector = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=21, padding=10),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 8, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    
    def forward(self, pred, target):
        # 使用目标谱进行可学习的峰检测
        peak_probability = self.peak_detector(target)
        
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # 自适应权重
        adaptive_weight = self.base_weight + (self.max_peak_weight - self.base_weight) * peak_probability
        
        # 加权损失
        weighted_loss = mse_loss * adaptive_weight
        base_loss = weighted_loss.mean()
        
        # 平滑性损失
        smoothness_loss = F.mse_loss(pred[:, :, 1:], pred[:, :, :-1], reduction='mean')
        
        total_loss = base_loss + self.smoothness_weight * smoothness_loss
        
        components = {
            'peak_loss': base_loss.item() * 0.7,  # 近似峰损失
            'compton_loss': base_loss.item() * 0.3,  # 近似康普顿损失
            'smoothness_loss': smoothness_loss.item(),
            'total_loss': total_loss.item(),
            'avg_peak_weight': adaptive_weight.mean().item()
        }
        
        # 使用适配器确保返回标准键
        adapted_components = LossComponentAdapter.adapt_components(components)
        # 保留额外的键
        adapted_components['avg_peak_weight'] = components['avg_peak_weight']
        
        return total_loss, adapted_components