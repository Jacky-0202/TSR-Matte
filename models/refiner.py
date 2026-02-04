# models/refiner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedInceptionBlock(nn.Module):
    """
    Inception-style block with Multi-Scale Dilation AND Spatial Smoothing.
    
    Structure:
        1. Multi-Scale Branches (1x1, 3x3 d=1, 3x3 d=2, 3x3 d=4)
        2. Fusion Layer (1x1 Mixing + 3x3 Smoothing) <--- UPGRADED
        3. Residual Connection
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 Conv (Pixel-wise)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Conv, Dilation=1 (Standard)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3 Conv, Dilation=2 (Medium)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 Conv, Dilation=4 (Large)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion Layer: [NEW] Enhanced with Spatial Smoothing
        self.fusion = nn.Sequential(
            # Step A: Channel Mixing (Compress concatenated features)
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Step B: Spatial Smoothing (Remove gridding artifacts from dilation)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # Note: No ReLU here because we add residual next
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Parallel Processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 2. Concatenation
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # 3. Fusion & Smoothing
        out = self.fusion(out)
        
        # 4. Residual Connection
        out += x
        
        return self.relu(out)

class Refiner(nn.Module):
    """
    Residual Refinement Module (RRM) with Inception Architecture.
    """
    def __init__(self, in_channels=4, mid_channels=64):
        super().__init__()
        
        # 1. Feature Extraction
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Refinement Body (Inception Blocks)
        # Using 2 blocks ensures deep enough feature mixing
        self.body = nn.Sequential(
            DilatedInceptionBlock(mid_channels, mid_channels),
            DilatedInceptionBlock(mid_channels, mid_channels)
        )
        
        # 3. Residual Prediction Head
        self.conv_out = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, img, coarse_alpha):
        x = torch.cat([img, coarse_alpha], dim=1)
        x = self.conv_in(x)
        x = self.body(x)
        residual = self.conv_out(x)
        
        return torch.sigmoid(coarse_alpha + residual)