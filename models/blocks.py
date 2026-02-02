# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Standard U-Net building block: (Conv3x3 -> BN -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Using Bilinear interpolation for smoothness (better for matting)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # x1: current feature (needs upsampling)
        # x2: skip connection from encoder
        x1 = self.up(x1)

        # Auto-padding to handle slight size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)