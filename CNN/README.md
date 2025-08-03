# LYSO闪烁体探测器分辨率优化系统

本项目基于一维深度卷积网络（1D-CNN），旨在将低分辨率的LYSO（硅酸钇镥）闪烁体探测器能谱，通过深度学习模型增强至接近高分辨率HPGe（高纯锗）探测器的能谱质量。

## 核心特性

- **先进的模型架构**:
    - **1D ResNet with SE**: 采用专门为一维能谱数据设计的残差网络（ResNet），并集成了Squeeze-and-Excitation (SE) 注意力机制，使模型能更好地关注关键能谱区域。
    - **U-Net Like Structure**: 改进版模型 (`ImprovedSpectralResNet1D`) 借鉴了U-Net的编码器-解码器结构和跳跃连接，有效融合多尺度特征。

- **极致的训练性能**:
    - **`torch.compile` 加速**: 在支持的PyTorch版本（2.0+）上自动启用，通过即时编译（JIT）大幅提升训练和推理速度。
    - **混合精度训练 (AMP)**: 内置支持FP16混合精度训练，显著降低显存占用，提升训练效率。
    - **梯度累积**: 完整支持梯度累积，允许在显存有限的硬件上模拟超大批次训练，探索更优的收敛路径。
    - **GPU原生实现**: 核心的损失函数和评估指标完全在GPU上计算，避免了CPU与GPU之间的数据传输瓶颈。

- **强大的功能与灵活性**:
    - **自定义复合损失函数**: 采用多种损失函数组合（如峰区域加权MSE、平滑度损失等），确保模型在提升分辨率的同时保持能谱的物理真实性。
    - **高效数据加载**: 支持多种数据格式（HDF5, LMDB），并实现了缓存、内存映射和流式加载等策略，从容应对百万级大规模数据集。
    - **分层配置系统**: 通过 `project_config.yaml` 控制高级任务流，通过 `configs/` 目录下的文件管理详细超参数，实现了清晰、灵活的实验管理。
    - **统一任务调度器**: `run_task.py` 脚本提供了覆盖数据准备、训练、推理、监控等全流程的统一入口。

## 项目结构

```
CNN/
├── run_task.py                  # 统一任务运行器
├── train_improved.py            # 主训练脚本 (已优化)
├── inference.py                 # 推理脚本
├── prepare_data.py              # 数据集分割工具
├── project_config.yaml          # 项目主任务配置文件
│
├── configs/
│   ├── default_config.yaml      # 默认超参数配置
│   └── rtx4060_config.yaml      # RTX 4060 优化配置
│
├── models/
│   ├── resnet1d.py              # 基础1D ResNet模型
│   └── resnet1d_improved.py     # 改进版模型 (带SE注意力)
│
├── utils/                       # 工具模块 (数据加载, 损失函数, 指标等)
├── data/                        # 存放原始数据 (H5, NPY等)
├── dataset_split/               # 存放分割后的数据集
├── checkpoints/                 # 存放模型检查点
├── logs/                        # 存放TensorBoard日志
└── README.md                    # 本文档
```

## 环境与安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd CNN

# 2. (可选) 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

## 快速开始 (端到端流程)

这是一个完整的、从数据准备到推理的流程示例。

```bash
# 步骤 1: 准备数据
# 将你的原始 .h5 或 .npy 数据文件放入 `data/` 目录
# 然后运行以下命令自动分割数据集
python prepare_data.py

# 步骤 2: 开始训练
# 使用为RTX 4060优化的配置进行训练
python train_improved.py --config configs/rtx4060_config.yaml

# 步骤 3: 监控训练 (在另一个终端中)
tensorboard --logdir logs

# 步骤 4: 使用训练好的最佳模型进行推理
python inference.py --checkpoint checkpoints/best_model.pth --input dataset_split/test/sample_0000.h5 --visualize
```

## 工作流与配置详解

本项目采用分层配置，易于管理和复现实验。

### 1. 主任务流配置 (`project_config.yaml`)

这个文件用于控制高级任务流程，例如数据分割、选择哪个配置文件进行训练、推理设置等。`run_task.py` 脚本会读取此文件来执行相应任务。

**示例: 使用 `run_task.py`**

```bash
# 根据 project_config.yaml 的设置，自动分割数据
python run_task.py prepare

# 根据 project_config.yaml 的设置，进行快速测试训练
python run_task.py train-quick

# 根据 project_config.yaml 的设置，进行完整训练
python run_task.py train-full

# 根据 project_config.yaml 的设置，运行推理
python run_task.py inference
```

### 2. 训练超参数配置 (`configs/*.yaml`)

`configs/` 目录下的文件定义了模型、训练、数据加载等所有详细的超参数。你可以创建多个配置文件来进行不同的实验。

**示例: `rtx4060_config.yaml`**
```yaml
model:
  name: ImprovedSpectralResNet1D # 使用带SE注意力的新模型
  num_blocks: 12
  channels: 64

training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 0.0005
  gradient_accumulation_steps: 4  # 有效批次 = 64 * 4 = 256
  mixed_precision: true           # 必须开启以利用硬件加速

data:
  format: h5
  train_path: dataset_split/train
  val_path: dataset_split/val
  num_workers: 0 # Windows下建议为0, Linux下可设为 4 或 8
```

**命令行覆盖**: 你可以在运行时动态覆盖配置参数。
```bash
python train_improved.py --config configs/rtx4060_config.yaml --batch_size 32 --learning_rate 0.0001
```

## 性能优化与硬件指南 (以RTX 4060为例)

### 显存不足 (CUDA Out of Memory)

如果遇到显存不足的问题，请按以下优先级顺序调整 `configs/rtx4060_config.yaml`:

1.  **增加梯度累积步数**:
    - `gradient_accumulation_steps: 4` -> `8` 或 `16`。这是最优先推荐的，因为它可以在不影响有效批次大小的情况下大幅降低显存。
2.  **减小批次大小**:
    - `batch_size: 64` -> `32` 或 `16`。
3.  **减小模型尺寸**:
    - `channels: 64` -> `48`。
    - `num_blocks: 12` -> `8`。

### 数据加载慢

- **大规模数据**: 对于百万级以上的数据集，建议先使用 `utils/data_preprocessor.py` 将其转换为 **LMDB** 格式。
- **硬件**: 确保数据集存放在 **SSD** 上。
- **工作进程**: 在Linux上，适当增加 `num_workers` (如 4, 8) 可以加速数据加载。在Windows上，由于多进程的限制，建议保持为 `0`。

## 故障排除

- **训练不稳定 (损失NaN或Inf)**:
    - 尝试减小学习率 `learning_rate`。
    - 检查 `gradient_clip` 参数是否设置在一个合理的值（如1.0）。
- **恢复训练**:
    - 使用 `--resume` 参数指定检查点路径即可无缝恢复训练。
    ```bash
    python train_improved.py --config <your_config> --resume checkpoints/checkpoint_epoch_50.pth
    ```

## 许可证

本项目采用 MIT 许可证。