import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ImprovedResidualBlock1D(nn.Module):
    """改进的残差块，支持下采样和上采样，并集成了SE注意力"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ImprovedResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)  # 添加SE模块
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 应用SE模块
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class ImprovedSpectralResNet1D(nn.Module):
    """改进的ResNet架构，支持多尺度特征和渐进式通道变化"""
    def __init__(self, input_channels=1, num_blocks=12, base_channels=64):
        super(ImprovedSpectralResNet1D, self).__init__()
        
        # 参数验证
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        
        # 初始卷积层
        self.initial_conv = nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 编码器路径（下采样）
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        channels = base_channels
        for i in range(num_blocks // 3):
            # 添加残差块
            self.encoder_blocks.append(ImprovedResidualBlock1D(channels, channels))
            
            # 每3个块后增加通道数并下采样
            if i < num_blocks // 3 - 1:
                next_channels = channels * 2
                downsample = nn.Sequential(
                    nn.Conv1d(channels, next_channels, kernel_size=1, stride=2),
                    nn.BatchNorm1d(next_channels)
                )
                self.downsample_layers.append(downsample)
                self.encoder_blocks.append(
                    ImprovedResidualBlock1D(channels, next_channels, stride=2, 
                                          downsample=downsample)
                )
                channels = next_channels
        
        # 中间块（瓶颈层）
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_blocks // 3):
            self.middle_blocks.append(ImprovedResidualBlock1D(channels, channels))
        
        # 解码器路径（上采样）
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(num_blocks // 3):
            if i > 0:
                # 上采样并减少通道数
                prev_channels = channels
                channels = channels // 2
                self.upsample_layers.append(
                    nn.ConvTranspose1d(prev_channels, channels, kernel_size=4, stride=2, padding=1)
                )
                self.decoder_blocks.append(ImprovedResidualBlock1D(channels, channels))
            else:
                self.decoder_blocks.append(ImprovedResidualBlock1D(channels, channels))
        
        # 输出层 - 渐进式通道减少
        self.output_layers = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, input_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 验证输入形状
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, channels, length), got {x.dim()}D")
        if x.size(1) != self.initial_conv.in_channels:
            raise ValueError(f"Expected {self.initial_conv.in_channels} input channels, got {x.size(1)}")
        
        # 初始处理
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)
        
        # 编码器路径
        encoder_features = []
        for i, block in enumerate(self.encoder_blocks):
            out = block(out)
            if i % 2 == 1 and i < len(self.encoder_blocks) - 1:  # 在下采样前保存特征
                encoder_features.append(out)
        
        # 中间块
        for block in self.middle_blocks:
            out = block(out)
        
        # 解码器路径
        decoder_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            if i > 0 and i - 1 < len(self.upsample_layers):
                # 上采样
                out = self.upsample_layers[i-1](out)
                # 跳跃连接（如果有对应的编码器特征）
                if decoder_idx < len(encoder_features):
                    encoder_feat = encoder_features[-(decoder_idx + 1)]
                    # 确保特征图大小匹配
                    if out.size(-1) != encoder_feat.size(-1):
                        diff = encoder_feat.size(-1) - out.size(-1)
                        if diff > 0:
                            out = F.pad(out, (diff // 2, diff - diff // 2))
                        else:
                            encoder_feat = F.pad(encoder_feat, (-diff // 2, -diff + diff // 2))
                    out = out + encoder_feat  # 跳跃连接
                    decoder_idx += 1
            out = block(out)
        
        # 输出处理
        out = self.output_layers(out)
        
        # 确保输出大小与输入匹配
        if out.size(-1) != x.size(-1):
            out = F.interpolate(out, size=x.size(-1), mode='linear', align_corners=False)
        
        return out