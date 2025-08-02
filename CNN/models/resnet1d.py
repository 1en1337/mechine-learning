import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class SpectralResNet1D(nn.Module):
    def __init__(self, input_channels=1, num_blocks=12, channels=64):
        super(SpectralResNet1D, self).__init__()
        
        # 参数验证
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        
        self.initial_conv = nn.Conv1d(input_channels, channels, kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(ResidualBlock1D(channels, channels))
        
        self.final_conv = nn.Conv1d(channels, input_channels, kernel_size=1)
        
    def forward(self, x):
        # 验证输入形状
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, channels, length), got {x.dim()}D")
        if x.size(1) != self.initial_conv.in_channels:
            raise ValueError(f"Expected {self.initial_conv.in_channels} input channels, got {x.size(1)}")
        
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)
        
        for block in self.res_blocks:
            out = block(out)
        
        out = self.final_conv(out)
        
        return out