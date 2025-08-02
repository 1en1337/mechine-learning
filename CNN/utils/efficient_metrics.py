import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class EfficientSpectralMetrics(nn.Module):
    """高效的PyTorch原生光谱指标系统 - 完全GPU实现，有状态累积"""
    
    def __init__(self, spectrum_length=4096, moving_average_window=0.9):
        super(EfficientSpectralMetrics, self).__init__()
        self.spectrum_length = spectrum_length
        self.moving_average_window = moving_average_window
        
        # 峰检测器（复用损失函数中的实现）
        self.register_buffer('peak_detector_kernel', self._create_peak_detector_kernel())
        
        # 状态缓冲区 - 使用register_buffer确保设备一致性
        self.register_buffer('total_fwhm_pred', torch.tensor(0.0))
        self.register_buffer('total_fwhm_target', torch.tensor(0.0))
        self.register_buffer('total_fwhm_improvement', torch.tensor(0.0))
        self.register_buffer('total_centroid_error', torch.tensor(0.0))
        self.register_buffer('total_peak_intensity_ratio', torch.tensor(0.0))
        self.register_buffer('total_spectral_correlation', torch.tensor(0.0))
        self.register_buffer('total_snr_improvement', torch.tensor(0.0))
        self.register_buffer('num_valid_samples', torch.tensor(0))
        
        # 移动平均指标
        self.register_buffer('ema_correlation', torch.tensor(0.0))
        self.register_buffer('ema_peak_ratio', torch.tensor(0.0))
        
    def _create_peak_detector_kernel(self, kernel_size=21):
        """创建峰检测核"""
        x = torch.linspace(-3, 3, kernel_size)
        mexican_hat = 2 / (torch.sqrt(torch.tensor(3.0)) * (torch.pi ** 0.25)) * \
                     (1 - x**2) * torch.exp(-0.5 * x**2)
        mexican_hat = mexican_hat / mexican_hat.abs().max()
        return mexican_hat.view(1, 1, -1)
    
    def gpu_peak_detection(self, spectrum):
        """GPU原生峰检测"""
        # 确保输入是3维张量 [batch, channel, length]
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
        # 峰响应 - 确保设备和数据类型匹配
        kernel = self.peak_detector_kernel.to(device=spectrum.device, dtype=spectrum.dtype)
        padding = kernel.size(-1) // 2
        peak_response = F.conv1d(spectrum, kernel, padding=padding)
        
        # 局部最大值
        max_pool = F.max_pool1d(peak_response, kernel_size=21, stride=1, padding=10)
        local_maxima = (peak_response == max_pool) & (peak_response > 0)
        
        # 动态阈值
        max_values = spectrum.max(dim=-1, keepdim=True)[0]
        threshold = max_values * 0.1
        significant_peaks = local_maxima & (spectrum > threshold)
        
        return significant_peaks, peak_response
    
    def estimate_fwhm_gpu(self, spectrum):
        """GPU上快速FWHM估算"""
        # 确保输入是3维张量 [batch, channel, length]
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
        batch_size = spectrum.shape[0]
        fwhms = []
        
        peaks, peak_response = self.gpu_peak_detection(spectrum)
        
        for i in range(batch_size):
            spec = spectrum[i, 0]  # [length]
            peak_mask = peaks[i, 0]  # [length]
            
            if peak_mask.sum() > 0:
                # 找到最强峰
                peak_indices = torch.nonzero(peak_mask, as_tuple=True)[0]
                if len(peak_indices) > 0:
                    peak_values = spec[peak_indices]
                    main_peak_idx = peak_indices[peak_values.argmax()]
                    
                    # 快速FWHM估算
                    peak_height = spec[main_peak_idx]
                    half_height = peak_height * 0.5
                    
                    # 向两侧搜索半高点
                    left_idx = main_peak_idx
                    right_idx = main_peak_idx
                    
                    # 向左搜索
                    while left_idx > 0 and spec[left_idx] > half_height:
                        left_idx -= 1
                    
                    # 向右搜索
                    while right_idx < len(spec) - 1 and spec[right_idx] > half_height:
                        right_idx += 1
                    
                    fwhm = (right_idx - left_idx).float()
                    fwhms.append(fwhm)
                else:
                    fwhms.append(torch.tensor(float('nan'), device=spectrum.device))
            else:
                fwhms.append(torch.tensor(float('nan'), device=spectrum.device))
        
        return torch.stack(fwhms)
    
    def estimate_centroid_gpu(self, spectrum):
        """GPU质心计算"""
        # 创建坐标网格
        coords = torch.arange(spectrum.shape[-1], device=spectrum.device, dtype=spectrum.dtype)
        coords = coords.view(1, 1, -1).expand_as(spectrum)
        
        # 加权平均计算质心
        weighted_sum = torch.sum(spectrum * coords, dim=-1)
        total_weight = torch.sum(spectrum, dim=-1)
        
        centroid = weighted_sum / (total_weight + 1e-8)
        return centroid.squeeze()
    
    def calculate_snr(self, spectrum):
        """计算信噪比"""
        # 估算信号：光谱最大值附近区域
        max_idx = spectrum.argmax(dim=-1)
        signal_region = []
        
        for i in range(spectrum.shape[0]):
            idx = max_idx[i]
            start = max(0, idx - 50)
            end = min(spectrum.shape[-1], idx + 50)
            # 确保start和end有效
            if start >= end:
                signal = torch.tensor(0.0, device=spectrum.device)
            else:
                signal = spectrum[i, :, start:end].mean()
            signal_region.append(signal)
        
        signal = torch.stack(signal_region)
        
        # 估算噪声：光谱首尾区域
        noise_start = spectrum[:, :, :200].std(dim=-1)
        noise_end = spectrum[:, :, -200:].std(dim=-1)
        noise = (noise_start + noise_end) / 2
        
        snr = signal / (noise.squeeze() + 1e-8)
        return snr
    
    def update(self, pred_spectrum, target_spectrum):
        """更新累积指标"""
        batch_size = pred_spectrum.shape[0]
        device = pred_spectrum.device
        
        # 计算FWHM
        fwhm_pred = self.estimate_fwhm_gpu(pred_spectrum)
        fwhm_target = self.estimate_fwhm_gpu(target_spectrum)
        
        # 过滤有效值
        valid_mask = ~(torch.isnan(fwhm_pred) | torch.isnan(fwhm_target))
        
        if valid_mask.sum() > 0:
            valid_fwhm_pred = fwhm_pred[valid_mask]
            valid_fwhm_target = fwhm_target[valid_mask]
            
            self.total_fwhm_pred += valid_fwhm_pred.sum()
            self.total_fwhm_target += valid_fwhm_target.sum()
            
            # FWHM改善
            # 避免除零
            safe_target = valid_fwhm_target + 1e-8
            improvement = (valid_fwhm_target - valid_fwhm_pred) / safe_target * 100
            self.total_fwhm_improvement += improvement.sum()
            
            self.num_valid_samples += valid_mask.sum()
        
        # 质心误差
        centroid_pred = self.estimate_centroid_gpu(pred_spectrum)
        centroid_target = self.estimate_centroid_gpu(target_spectrum)
        centroid_error = torch.abs(centroid_pred - centroid_target).mean()
        self.total_centroid_error += centroid_error
        
        # 峰强度比
        pred_peak_intensity = pred_spectrum.max(dim=-1)[0].mean()
        target_peak_intensity = target_spectrum.max(dim=-1)[0].mean()
        peak_ratio = pred_peak_intensity / (target_peak_intensity + 1e-8)
        self.total_peak_intensity_ratio += peak_ratio
        
        # 光谱相关性
        pred_norm = F.normalize(pred_spectrum.view(batch_size, -1), dim=1)
        target_norm = F.normalize(target_spectrum.view(batch_size, -1), dim=1)
        correlation = (pred_norm * target_norm).sum(dim=1).mean()
        self.total_spectral_correlation += correlation
        
        # 信噪比改善
        snr_pred = self.calculate_snr(pred_spectrum).mean()
        snr_target = self.calculate_snr(target_spectrum).mean()
        snr_improvement = (snr_pred - snr_target) / (snr_target + 1e-8) * 100
        self.total_snr_improvement += snr_improvement
        
        # 更新移动平均
        if self.ema_correlation == 0:
            self.ema_correlation = correlation
            self.ema_peak_ratio = peak_ratio
        else:
            self.ema_correlation = self.moving_average_window * self.ema_correlation + \
                                 (1 - self.moving_average_window) * correlation
            self.ema_peak_ratio = self.moving_average_window * self.ema_peak_ratio + \
                                (1 - self.moving_average_window) * peak_ratio
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        if self.num_valid_samples > 0:
            # 计算平均值
            avg_fwhm_pred = self.total_fwhm_pred / self.num_valid_samples
            avg_fwhm_target = self.total_fwhm_target / self.num_valid_samples
            avg_fwhm_improvement = self.total_fwhm_improvement / self.num_valid_samples
            avg_centroid_error = self.total_centroid_error / self.num_valid_samples
            avg_peak_ratio = self.total_peak_intensity_ratio / self.num_valid_samples
            avg_correlation = self.total_spectral_correlation / self.num_valid_samples
            avg_snr_improvement = self.total_snr_improvement / self.num_valid_samples
            
            return {
                'fwhm_pred': avg_fwhm_pred.item(),
                'fwhm_target': avg_fwhm_target.item(),
                'fwhm_improvement_pct': avg_fwhm_improvement.item(),
                'centroid_error': avg_centroid_error.item(),
                'peak_intensity_ratio': avg_peak_ratio.item(),
                'spectral_correlation': avg_correlation.item(),
                'snr_improvement_pct': avg_snr_improvement.item(),
                'ema_correlation': self.ema_correlation.item(),
                'ema_peak_ratio': self.ema_peak_ratio.item(),
                'num_valid_samples': self.num_valid_samples.item()
            }
        else:
            return {
                'fwhm_pred': 0.0,
                'fwhm_target': 0.0,
                'fwhm_improvement_pct': 0.0,
                'centroid_error': 0.0,
                'peak_intensity_ratio': 0.0,
                'spectral_correlation': 0.0,
                'snr_improvement_pct': 0.0,
                'ema_correlation': 0.0,
                'ema_peak_ratio': 0.0,
                'num_valid_samples': 0
            }
    
    def reset(self):
        """重置所有累积状态"""
        self.total_fwhm_pred.zero_()
        self.total_fwhm_target.zero_()
        self.total_fwhm_improvement.zero_()
        self.total_centroid_error.zero_()
        self.total_peak_intensity_ratio.zero_()
        self.total_spectral_correlation.zero_()
        self.total_snr_improvement.zero_()
        self.num_valid_samples.zero_()
        # 不重置移动平均，保持连续性
    
    def get_current_batch_metrics(self, pred_spectrum, target_spectrum) -> Dict[str, float]:
        """获取当前批次的即时指标（不累积）"""
        with torch.no_grad():
            batch_size = pred_spectrum.shape[0]
            
            # 光谱相关性
            pred_norm = F.normalize(pred_spectrum.view(batch_size, -1), dim=1)
            target_norm = F.normalize(target_spectrum.view(batch_size, -1), dim=1)
            correlation = (pred_norm * target_norm).sum(dim=1).mean()
            
            # 峰强度比
            pred_peak = pred_spectrum.max(dim=-1)[0].mean()
            target_peak = target_spectrum.max(dim=-1)[0].mean()
            peak_ratio = pred_peak / (target_peak + 1e-8)
            
            # MSE
            mse = F.mse_loss(pred_spectrum, target_spectrum)
            
            return {
                'batch_correlation': correlation.item(),
                'batch_peak_ratio': peak_ratio.item(),
                'batch_mse': mse.item()
            }


class FastTrainingMetrics(nn.Module):
    """轻量级训练时指标 - 最小化计算开销"""
    
    def __init__(self, update_frequency=10):
        super(FastTrainingMetrics, self).__init__()
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # 轻量级状态
        self.register_buffer('running_correlation', torch.tensor(0.0))
        self.register_buffer('running_peak_ratio', torch.tensor(0.0))
        self.register_buffer('running_mse', torch.tensor(0.0))
        self.register_buffer('num_updates', torch.tensor(0))
    
    def update(self, pred_spectrum, target_spectrum):
        """快速更新 - 只计算最基本的指标"""
        self.step_count += 1
        
        # 只在指定频率更新，减少计算
        if self.step_count % self.update_frequency == 0:
            batch_size = pred_spectrum.shape[0]
            
            # 归一化相关性（最重要的指标）
            pred_norm = F.normalize(pred_spectrum.view(batch_size, -1), dim=1)
            target_norm = F.normalize(target_spectrum.view(batch_size, -1), dim=1)
            correlation = (pred_norm * target_norm).sum(dim=1).mean()
            
            # 峰强度比
            pred_peak = pred_spectrum.max(dim=-1)[0].mean()
            target_peak = target_spectrum.max(dim=-1)[0].mean()
            peak_ratio = pred_peak / (target_peak + 1e-8)
            
            # MSE
            mse = F.mse_loss(pred_spectrum, target_spectrum)
            
            # 更新运行平均
            self.num_updates += 1
            alpha = 1.0 / self.num_updates
            
            self.running_correlation = (1 - alpha) * self.running_correlation + alpha * correlation
            self.running_peak_ratio = (1 - alpha) * self.running_peak_ratio + alpha * peak_ratio
            self.running_mse = (1 - alpha) * self.running_mse + alpha * mse
    
    def compute(self) -> Dict[str, float]:
        """获取当前运行指标"""
        return {
            'running_correlation': self.running_correlation.item(),
            'running_peak_ratio': self.running_peak_ratio.item(),
            'running_mse': self.running_mse.item(),
            'step_count': self.step_count
        }
    
    def reset(self):
        """重置"""
        self.running_correlation.zero_()
        self.running_peak_ratio.zero_()
        self.running_mse.zero_()
        self.num_updates.zero_()
        self.step_count = 0


# 为了向后兼容，包装原有的SpectralMetrics
class OptimizedSpectralMetrics:
    """优化版本的SpectralMetrics类，提供与原版相同的接口"""
    
    def __init__(self):
        self.efficient_metrics = None
    
    def _ensure_metrics_initialized(self, device):
        """确保指标在正确的设备上初始化"""
        if self.efficient_metrics is None:
            self.efficient_metrics = EfficientSpectralMetrics().to(device)
    
    @staticmethod
    def compute_all_metrics(pred_spectrum, target_spectrum, original_spectrum=None):
        """
        兼容原有接口的静态方法，但内部使用GPU优化版本
        注意：这个方法仍然是无状态的，主要用于推理和最终评估
        """
        device = pred_spectrum.device
        metrics_calculator = EfficientSpectralMetrics().to(device)
        
        # 确保输入是正确的形状
        if len(pred_spectrum.shape) == 1:
            pred_spectrum = pred_spectrum.unsqueeze(0).unsqueeze(0)
        elif len(pred_spectrum.shape) == 2:
            pred_spectrum = pred_spectrum.unsqueeze(1)
            
        if len(target_spectrum.shape) == 1:
            target_spectrum = target_spectrum.unsqueeze(0).unsqueeze(0)
        elif len(target_spectrum.shape) == 2:
            target_spectrum = target_spectrum.unsqueeze(1)
        
        # 计算当前批次指标
        batch_metrics = metrics_calculator.get_current_batch_metrics(pred_spectrum, target_spectrum)
        
        # 估算FWHM
        fwhm_pred = metrics_calculator.estimate_fwhm_gpu(pred_spectrum)
        fwhm_target = metrics_calculator.estimate_fwhm_gpu(target_spectrum)
        
        # 组织返回格式以兼容原有代码
        metrics = {}
        
        if not torch.isnan(fwhm_pred[0]) and not torch.isnan(fwhm_target[0]):
            metrics['fwhm_pred'] = fwhm_pred[0].item()
            metrics['fwhm_target'] = fwhm_target[0].item()
            metrics['fwhm_improvement'] = ((fwhm_target[0] - fwhm_pred[0]) / fwhm_target[0] * 100).item()
        
        if original_spectrum is not None:
            if len(original_spectrum.shape) == 1:
                original_spectrum = original_spectrum.unsqueeze(0).unsqueeze(0)
            elif len(original_spectrum.shape) == 2:
                original_spectrum = original_spectrum.unsqueeze(1)
            
            fwhm_original = metrics_calculator.estimate_fwhm_gpu(original_spectrum)
            if not torch.isnan(fwhm_original[0]) and not torch.isnan(fwhm_pred[0]):
                metrics['fwhm_original'] = fwhm_original[0].item()
                metrics['fwhm_reduction'] = ((fwhm_original[0] - fwhm_pred[0]) / fwhm_original[0] * 100).item()
        
        # 质心
        centroid_pred = metrics_calculator.estimate_centroid_gpu(pred_spectrum)
        centroid_target = metrics_calculator.estimate_centroid_gpu(target_spectrum)
        
        if len(centroid_pred) > 0 and len(centroid_target) > 0:
            metrics['centroid_pred'] = centroid_pred[0].item()
            metrics['centroid_target'] = centroid_target[0].item()
            metrics['centroid_shift'] = abs(centroid_pred[0] - centroid_target[0]).item()
        else:
            metrics['centroid_pred'] = 0.0
            metrics['centroid_target'] = 0.0
            metrics['centroid_shift'] = 0.0
        
        # 添加批次指标
        metrics.update(batch_metrics)
        
        return metrics