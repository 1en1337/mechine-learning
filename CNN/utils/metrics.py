import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))


def calculate_fwhm(spectrum, channel_axis=None):
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()
    
    # 处理不同维度的输入
    if len(spectrum.shape) == 3:
        spectrum = spectrum[0, 0, :]
    elif len(spectrum.shape) == 2:
        spectrum = spectrum[0, :]
    elif len(spectrum.shape) == 1:
        pass  # 已经是一维
    else:
        return None, None
    
    peaks, properties = find_peaks(spectrum, height=np.max(spectrum)*0.3, distance=50)
    
    if len(peaks) == 0:
        return None, None
    
    main_peak_idx = peaks[np.argmax(spectrum[peaks])]
    peak_height = spectrum[main_peak_idx]
    half_height = peak_height / 2
    
    # FWHM计算 - 使用平滑后的光谱避免噪声干扰
    from scipy.ndimage import gaussian_filter1d
    smoothed_spectrum = gaussian_filter1d(spectrum, sigma=2)
    
    left_idx = main_peak_idx
    while left_idx > 0 and smoothed_spectrum[left_idx] > half_height:
        left_idx -= 1
    
    right_idx = main_peak_idx
    while right_idx < len(spectrum) - 1 and smoothed_spectrum[right_idx] > half_height:
        right_idx += 1
    
    if left_idx > 0 and right_idx < len(spectrum) - 1:
        # 确保索引不越界
        start_idx = max(0, left_idx - 10)
        end_idx = min(len(spectrum), right_idx + 10)
        x = np.arange(start_idx, end_idx)
        y = spectrum[start_idx:end_idx]
        
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=[peak_height, main_peak_idx, 10])
            fwhm = 2.355 * abs(popt[2])
            return fwhm, main_peak_idx
        except (RuntimeError, ValueError):
            fwhm = right_idx - left_idx
            return fwhm, main_peak_idx
    
    return None, None


def calculate_peak_centroid(spectrum):
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()
    
    # 处理不同维度的输入
    if len(spectrum.shape) == 3:
        spectrum = spectrum[0, 0, :]
    elif len(spectrum.shape) == 2:
        spectrum = spectrum[0, :]
    elif len(spectrum.shape) == 1:
        pass  # 已经是一维
    else:
        return None
    
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.3, distance=50)
    
    if len(peaks) == 0:
        return None
    
    main_peak_idx = peaks[np.argmax(spectrum[peaks])]
    
    window = 20
    start = max(0, main_peak_idx - window)
    end = min(len(spectrum), main_peak_idx + window)
    
    x = np.arange(start, end)
    y = spectrum[start:end]
    
    centroid = np.sum(x * y) / np.sum(y)
    
    return centroid


def calculate_peak_to_compton(spectrum):
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()
    
    # 处理不同维度的输入
    if len(spectrum.shape) == 3:
        spectrum = spectrum[0, 0, :]
    elif len(spectrum.shape) == 2:
        spectrum = spectrum[0, :]
    elif len(spectrum.shape) == 1:
        pass  # 已经是一维
    else:
        return None
    
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.3, distance=50)
    
    if len(peaks) == 0:
        return None
    
    main_peak_idx = peaks[np.argmax(spectrum[peaks])]
    peak_value = spectrum[main_peak_idx]
    
    # 动态计算康普顿区域 - 考虑峰的位置
    spectrum_length = len(spectrum)
    # 康普顿边缘通常在峰位置的0.3-0.7倍，但需要确保在光谱范围内
    compton_region_start = max(0, int(main_peak_idx * 0.3))
    compton_region_end = min(int(main_peak_idx * 0.7), main_peak_idx - 50)
    
    # 确保康普顿区域有意义
    if compton_region_end > compton_region_start + 10:  # 至少有10个通道
        compton_value = np.mean(spectrum[compton_region_start:compton_region_end])
        peak_to_compton = peak_value / (compton_value + 1e-8)
        return peak_to_compton
    
    return None


class SpectralMetrics:
    @staticmethod
    def compute_all_metrics(pred_spectrum, target_spectrum, original_spectrum=None):
        metrics = {}
        
        fwhm_pred, peak_idx_pred = calculate_fwhm(pred_spectrum)
        fwhm_target, peak_idx_target = calculate_fwhm(target_spectrum)
        
        if fwhm_pred is not None and fwhm_target is not None and fwhm_target > 0:
            metrics['fwhm_pred'] = fwhm_pred
            metrics['fwhm_target'] = fwhm_target
            metrics['fwhm_improvement'] = (fwhm_target - fwhm_pred) / fwhm_target * 100
        
        if original_spectrum is not None:
            fwhm_original, _ = calculate_fwhm(original_spectrum)
            if fwhm_original is not None and fwhm_pred is not None and fwhm_original > 0:
                metrics['fwhm_original'] = fwhm_original
                metrics['fwhm_reduction'] = (fwhm_original - fwhm_pred) / fwhm_original * 100
        
        centroid_pred = calculate_peak_centroid(pred_spectrum)
        centroid_target = calculate_peak_centroid(target_spectrum)
        
        if centroid_pred is not None and centroid_target is not None:
            metrics['centroid_pred'] = centroid_pred
            metrics['centroid_target'] = centroid_target
            metrics['centroid_shift'] = abs(centroid_pred - centroid_target)
        
        ptc_pred = calculate_peak_to_compton(pred_spectrum)
        ptc_target = calculate_peak_to_compton(target_spectrum)
        
        if ptc_pred is not None and ptc_target is not None and ptc_target > 0:
            metrics['peak_to_compton_pred'] = ptc_pred
            metrics['peak_to_compton_target'] = ptc_target
            metrics['ptc_improvement'] = (ptc_pred - ptc_target) / ptc_target * 100
        
        return metrics


class PreciseSpectralMetrics:
    """
    精确的光谱指标计算类 - 保留原有SciPy实现用于最终评估和研究分析
    注意：此类主要用于推理和最终结果评估，不建议在训练循环中使用
    """
    
    @staticmethod
    def compute_precise_fwhm(spectrum, use_gaussian_fit=True):
        """精确FWHM计算，使用高斯拟合"""
        if isinstance(spectrum, torch.Tensor):
            spectrum = spectrum.detach().cpu().numpy()
        
        if len(spectrum.shape) >= 2:
            spectrum = spectrum.squeeze()
        
        try:
            # 使用原有的精确方法
            fwhm, peak_idx = calculate_fwhm(spectrum)
            
            if use_gaussian_fit and fwhm is not None:
                # 额外的高斯拟合验证
                peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*0.3, distance=50)
                if len(peaks) > 0:
                    main_peak_idx = peaks[np.argmax(spectrum[peaks])]
                    
                    # 扩展窗口进行更精确的高斯拟合
                    window = min(100, len(spectrum) // 10)
                    start = max(0, main_peak_idx - window)
                    end = min(len(spectrum), main_peak_idx + window)
                    
                    x_fit = np.arange(start, end)
                    y_fit = spectrum[start:end]
                    
                    try:
                        from scipy.optimize import curve_fit
                        popt, _ = curve_fit(
                            gaussian, x_fit, y_fit, 
                            p0=[np.max(y_fit), main_peak_idx, 10],
                            maxfev=1000
                        )
                        fitted_fwhm = 2.355 * abs(popt[2])
                        
                        # 如果拟合结果合理，使用拟合值
                        if 0.5 <= fitted_fwhm <= len(spectrum) * 0.1:
                            return fitted_fwhm, main_peak_idx
                    except:
                        pass
            
            return fwhm, peak_idx
            
        except Exception as e:
            print(f"精确FWHM计算失败: {e}")
            return None, None
    
    @staticmethod
    def compute_detailed_metrics(pred_spectrum, target_spectrum, original_spectrum=None):
        """
        计算详细的光谱质量指标，用于深度分析
        """
        metrics = {}
        
        # 基础指标
        basic_metrics = SpectralMetrics.compute_all_metrics(
            pred_spectrum, target_spectrum, original_spectrum
        )
        metrics.update(basic_metrics)
        
        # 精确FWHM
        pred_fwhm, _ = PreciseSpectralMetrics.compute_precise_fwhm(pred_spectrum)
        target_fwhm, _ = PreciseSpectralMetrics.compute_precise_fwhm(target_spectrum)
        
        if pred_fwhm is not None and target_fwhm is not None:
            metrics['precise_fwhm_pred'] = pred_fwhm
            metrics['precise_fwhm_target'] = target_fwhm
            metrics['precise_fwhm_improvement'] = (target_fwhm - pred_fwhm) / target_fwhm * 100
        
        # 频域分析
        try:
            if isinstance(pred_spectrum, torch.Tensor):
                pred_np = pred_spectrum.detach().cpu().numpy().squeeze()
                target_np = target_spectrum.detach().cpu().numpy().squeeze()
            else:
                pred_np = pred_spectrum.squeeze()
                target_np = target_spectrum.squeeze()
            
            # FFT分析
            pred_fft = np.fft.fft(pred_np)
            target_fft = np.fft.fft(target_np)
            
            # 频域相关性
            freq_correlation = np.corrcoef(np.abs(pred_fft), np.abs(target_fft))[0, 1]
            metrics['frequency_correlation'] = freq_correlation
            
            # 高频噪声分析
            freq_len_pred = len(pred_fft)
            freq_len_target = len(target_fft)
            high_freq_pred = np.abs(pred_fft[freq_len_pred//2:]) if freq_len_pred > 0 else np.array([0])
            high_freq_target = np.abs(target_fft[freq_len_target//2:]) if freq_len_target > 0 else np.array([0])
            
            target_mean = np.mean(high_freq_target)
            if target_mean > 1e-8:
                noise_reduction = (target_mean - np.mean(high_freq_pred)) / target_mean * 100
            else:
                noise_reduction = 0.0
            metrics['noise_reduction_pct'] = noise_reduction
            
        except Exception as e:
            print(f"频域分析失败: {e}")
        
        # 峰形状分析
        try:
            # 主峰对称性
            peaks_pred, _ = find_peaks(pred_np, height=np.max(pred_np)*0.5, distance=50)
            peaks_target, _ = find_peaks(target_np, height=np.max(target_np)*0.5, distance=50)
            
            if len(peaks_pred) > 0 and len(peaks_target) > 0:
                main_peak_pred = peaks_pred[np.argmax(pred_np[peaks_pred])]
                main_peak_target = peaks_target[np.argmax(target_np[peaks_target])]
                
                # 峰对称性（左右半高宽比）
                def peak_symmetry(spectrum, peak_idx):
                    peak_height = spectrum[peak_idx]
                    half_height = peak_height * 0.5
                    
                    left_idx = peak_idx
                    while left_idx > 0 and spectrum[left_idx] > half_height:
                        left_idx -= 1
                    
                    right_idx = peak_idx
                    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_height:
                        right_idx += 1
                    
                    left_width = peak_idx - left_idx
                    right_width = right_idx - peak_idx
                    
                    if right_width > 0:
                        return left_width / right_width
                    return float('inf')
                
                sym_pred = peak_symmetry(pred_np, main_peak_pred)
                sym_target = peak_symmetry(target_np, main_peak_target)
                
                metrics['peak_symmetry_pred'] = sym_pred
                metrics['peak_symmetry_target'] = sym_target
                
        except Exception as e:
            print(f"峰形状分析失败: {e}")
        
        return metrics
    
    @staticmethod
    def generate_analysis_report(metrics_dict, save_path=None):
        """生成详细的分析报告"""
        report = []
        report.append("=" * 50)
        report.append("光谱增强详细分析报告")
        report.append("=" * 50)
        
        # FWHM分析
        if 'fwhm_improvement' in metrics_dict:
            report.append(f"\n📊 分辨率改善分析:")
            report.append(f"  预测FWHM: {metrics_dict.get('fwhm_pred', 'N/A'):.3f}")
            report.append(f"  目标FWHM: {metrics_dict.get('fwhm_target', 'N/A'):.3f}")
            report.append(f"  改善百分比: {metrics_dict['fwhm_improvement']:.2f}%")
            
            if 'precise_fwhm_improvement' in metrics_dict:
                report.append(f"  精确改善百分比: {metrics_dict['precise_fwhm_improvement']:.2f}%")
        
        # 光谱质量
        if 'spectral_correlation' in metrics_dict:
            report.append(f"\n📈 光谱质量分析:")
            report.append(f"  光谱相关性: {metrics_dict['spectral_correlation']:.4f}")
            
            if 'frequency_correlation' in metrics_dict:
                report.append(f"  频域相关性: {metrics_dict['frequency_correlation']:.4f}")
            
            if 'noise_reduction_pct' in metrics_dict:
                report.append(f"  噪声减少: {metrics_dict['noise_reduction_pct']:.2f}%")
        
        # 峰特性
        if 'peak_intensity_ratio' in metrics_dict:
            report.append(f"\n🎯 峰特性分析:")
            report.append(f"  峰强度比: {metrics_dict['peak_intensity_ratio']:.3f}")
            report.append(f"  质心偏移: {metrics_dict.get('centroid_shift', 'N/A'):.2f}")
            
            if 'peak_symmetry_pred' in metrics_dict:
                report.append(f"  峰对称性(预测): {metrics_dict['peak_symmetry_pred']:.3f}")
                report.append(f"  峰对称性(目标): {metrics_dict['peak_symmetry_target']:.3f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"分析报告已保存至: {save_path}")
        
        return report_text