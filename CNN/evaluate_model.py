import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from models.resnet1d import SpectralResNet1D
from models.resnet1d_improved import ImprovedSpectralResNet1D
from utils.dataset import create_data_loaders
from utils.efficient_metrics import EfficientSpectralMetrics
from utils.checkpoint_manager import CheckpointManager
from utils.visualization import plot_spectra_comparison

class Evaluator:
    """
    模型评估器，用于在测试集上对训练好的模型进行详细评估。
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """加载模型和配置"""
        checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model_name = config.get('model', {}).get('name', 'SpectralResNet1D')
        model_config = config.get('model', {})

        if model_name == 'ImprovedSpectralResNet1D':
            self.model = ImprovedSpectralResNet1D(
                input_channels=model_config.get('input_channels', 1),
                num_blocks=model_config.get('num_blocks', 12),
                base_channels=model_config.get('channels', 64)
            )
        else:
            self.model = SpectralResNet1D(
                input_channels=model_config.get('input_channels', 1),
                num_blocks=model_config.get('num_blocks', 12),
                channels=model_config.get('channels', 64)
            )
        
        # 使用CheckpointManager安全加载
        manager = CheckpointManager()
        manager._load_model_state(checkpoint, self.model, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(torch, 'compile'):
            print("Compiling the model for evaluation...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def setup_dataloader(self):
        """设置测试数据加载器"""
        # 这里的val_loader实际上是我们的test_loader
        _, self.test_loader = create_data_loaders(
            train_path=self.args.test_dir, # 伪造一个训练路径
            val_path=self.args.test_dir,
            batch_size=self.args.batch_size,
            num_workers=0 # 在评估时通常使用0
        )

    def run(self):
        """执行评估"""
        self.load_model()
        self.setup_dataloader()
        
        metrics = EfficientSpectralMetrics().to(self.device)
        all_samples_metrics = []

        with torch.no_grad():
            for i, (lyso, hpge) in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                lyso = lyso.to(self.device)
                hpge = hpge.to(self.device)
                
                pred = self.model(lyso)
                
                # 累积总体指标
                metrics.update(pred, hpge)
                
                # 计算并保存每个样本的指标以供排序
                for j in range(lyso.size(0)):
                    sample_corr = torch.nn.functional.cosine_similarity(
                        pred[j].flatten(), hpge[j].flatten(), dim=0
                    ).item()
                    
                    all_samples_metrics.append({
                        'correlation': sample_corr,
                        'lyso': lyso[j].cpu().numpy(),
                        'hpge': hpge[j].cpu().numpy(),
                        'pred': pred[j].cpu().numpy(),
                        'filename': self.test_loader.dataset.file_list[i * self.args.batch_size + j].name
                    })

        # 计算最终平均指标
        final_metrics = metrics.compute()
        
        # 生成并保存报告
        self.generate_report(final_metrics)
        
        # 可视化最佳和最差样本
        self.visualize_samples(all_samples_metrics)

    def generate_report(self, final_metrics):
        """生成并打印/保存评估报告"""
        report = f"""
# 模型评估报告

**模型检查点**: {self.args.checkpoint}
**测试数据集**: {self.args.test_dir}
**评估样本数**: {len(self.test_loader.dataset)}

---

## **总体性能指标**

| 指标 (Metric) | 数值 (Value) | 描述 |
| :--- | :--- | :--- |
| **FWHM 改善率 (%)** | `{final_metrics.get('fwhm_improvement_pct', 0):.2f}` | **核心指标**: 预测谱相对于目标谱的半峰全宽（分辨率）的平均改善百分比。越高越好。 |
| **光谱相关性** | `{final_metrics.get('spectral_correlation', 0):.4f}` | 预测谱与目标谱的余弦相似度。越接近1，形状越相似。 |
| **峰强度比** | `{final_metrics.get('peak_intensity_ratio', 0):.4f}` | 预测谱主峰高度与目标谱主峰高度的比值。越接近1，峰高重建越准。 |
| **质心误差 (channel)** | `{final_metrics.get('centroid_error', 0):.2f}` | 预测谱与目标谱峰位中心的平均绝对误差（单位：道）。越小越好。 |
| **信噪比改善率 (%)** | `{final_metrics.get('snr_improvement_pct', 0):.2f}` | 预测谱相对于目标谱的信噪比改善。越高越好。 |

---
"""
        print(report)
        with open(self.output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"评估报告已保存至: {self.output_dir / 'evaluation_report.md'}")

    def visualize_samples(self, all_samples_metrics):
        """可视化最佳和最差的N个样本"""
        if self.args.num_samples_to_visualize == 0:
            return
            
        # 按相关性排序
        all_samples_metrics.sort(key=lambda x: x['correlation'], reverse=True)
        
        best_samples = all_samples_metrics[:self.args.num_samples_to_visualize]
        worst_samples = all_samples_metrics[-self.args.num_samples_to_visualize:]
        
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        print(f"正在保存 {len(best_samples)} 个最佳样本和 {len(worst_samples)} 个最差样本的可视化结果...")
        
        for i, sample in enumerate(best_samples):
            plot_spectra_comparison(
                sample['lyso'].squeeze(),
                sample['hpge'].squeeze(),
                sample['pred'].squeeze(),
                title=f"Best Sample #{i+1} | Correlation: {sample['correlation']:.4f}\nFile: {sample['filename']}",
                save_path=vis_dir / f"best_sample_{i+1}.png"
            )
            
        for i, sample in enumerate(worst_samples):
            plot_spectra_comparison(
                sample['lyso'].squeeze(),
                sample['hpge'].squeeze(),
                sample['pred'].squeeze(),
                title=f"Worst Sample #{i+1} | Correlation: {sample['correlation']:.4f}\nFile: {sample['filename']}",
                save_path=vis_dir / f"worst_sample_{i+1}.png"
            )
        
        print(f"可视化结果已保存至: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="模型效果评估脚本")
    parser.add_argument('--checkpoint', type=str, required=True, help='要评估的模型检查点路径')
    parser.add_argument('--test_dir', type=str, required=True, help='测试数据集的目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='存放评估报告和可视化结果的目录')
    parser.add_argument('--batch_size', type=int, default=64, help='评估时使用的批次大小')
    parser.add_argument('--num_samples_to_visualize', type=int, default=5, help='要保存的最佳/最差样本的可视化数量')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(args)
    evaluator.run()

if __name__ == '__main__':
    main()