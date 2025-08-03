import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import seaborn as sns
from .metrics import calculate_fwhm, calculate_peak_centroid, calculate_peak_to_compton


class SpectralVisualizer:
    def __init__(self, save_dir='results/figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except (OSError, ValueError):
            try:
                plt.style.use('seaborn-darkgrid')
            except (OSError, ValueError):
                try:
                    plt.style.use('seaborn')
                except (OSError, ValueError):
                    pass  # 使用默认样式
        
    def plot_spectrum_comparison(self, lyso, hpge, pred, title="Spectrum Comparison", save_name=None):
        if isinstance(lyso, torch.Tensor):
            lyso = lyso.detach().cpu().numpy()
        if isinstance(hpge, torch.Tensor):
            hpge = hpge.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        if len(lyso.shape) == 3:
            lyso = lyso[0, 0, :]
            hpge = hpge[0, 0, :]
            pred = pred[0, 0, :]
        elif len(lyso.shape) == 2:
            lyso = lyso[0, :]
            hpge = hpge[0, :]
            pred = pred[0, :]
        
        channels = np.arange(len(lyso))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(channels, lyso, 'b-', label='LYSO (Input)', alpha=0.7, linewidth=1.5)
        ax1.plot(channels, pred, 'r-', label='CNN Enhanced', alpha=0.8, linewidth=2)
        ax1.plot(channels, hpge, 'g--', label='HPGe (Target)', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Counts (normalized)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        start = 1000
        end = 2000
        ax2.plot(channels[start:end], lyso[start:end], 'b-', label='LYSO (Input)', alpha=0.7, linewidth=1.5)
        ax2.plot(channels[start:end], pred[start:end], 'r-', label='CNN Enhanced', alpha=0.8, linewidth=2)
        ax2.plot(channels[start:end], hpge[start:end], 'g--', label='HPGe (Target)', alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Counts (normalized)')
        ax2.set_title('Zoomed View (Channels 1000-2000)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # 释放内存
        
    def plot_metrics_comparison(self, lyso, hpge, pred, save_name=None):
        fwhm_lyso, peak_lyso = calculate_fwhm(lyso)
        fwhm_pred, peak_pred = calculate_fwhm(pred)
        fwhm_hpge, peak_hpge = calculate_fwhm(hpge)
        
        ptc_lyso = calculate_peak_to_compton(lyso)
        ptc_pred = calculate_peak_to_compton(pred)
        ptc_hpge = calculate_peak_to_compton(hpge)
        
        metrics = {
            'FWHM': [fwhm_lyso, fwhm_pred, fwhm_hpge],
            'Peak-to-Compton': [ptc_lyso, ptc_pred, ptc_hpge]
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        categories = ['LYSO\n(Input)', 'CNN\n(Enhanced)', 'HPGe\n(Target)']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
            
            for bar, value in zip(bars, values):
                if value is not None:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # 释放内存
        
    def plot_loss_curves(self, train_losses, val_losses, save_name='loss_curves.png'):
        epochs = range(1, len(train_losses) + 1)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_residuals(self, hpge, pred, save_name='residuals.png'):
        if isinstance(hpge, torch.Tensor):
            hpge = hpge.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        
        if len(hpge.shape) == 3:
            hpge = hpge[0, 0, :]
            pred = pred[0, 0, :]
        elif len(hpge.shape) == 2:
            hpge = hpge[0, :]
            pred = pred[0, :]
        
        residuals = hpge - pred
        channels = np.arange(len(residuals))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(channels, residuals, 'k-', alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Residual (Target - Predicted)')
        ax1.set_title('Residual Plot')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(residuals, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax2.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_residual:.4f}')
        ax2.axvline(mean_residual + std_residual, color='orange', linestyle='--', linewidth=1, label=f'±1σ: {std_residual:.4f}')
        ax2.axvline(mean_residual - std_residual, color='orange', linestyle='--', linewidth=1)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()