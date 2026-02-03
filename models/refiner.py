# models/refiner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedInceptionBlock(nn.Module):
    """
    Inception-style block with Multi-Scale Dilation.
    
    Instead of a simple sequential convolution, this block captures context 
    at multiple scales simultaneously. This is crucial for Matting because 
    edge uncertainty varies in size (e.g., thin hair vs. wide motion blur).
    
    Structure:
        - Branch 1: 1x1 Conv (Local detail preservation)
        - Branch 2: 3x3 Conv, Dilation 1 (Standard field of view)
        - Branch 3: 3x3 Conv, Dilation 2 (Medium field of view)
        - Branch 4: 3x3 Conv, Dilation 4 (Large field of view for wide halos)
        - Fusion:   Concatenates all branches and fuses them via 1x1 Conv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # We split the output channels among 4 branches to keep parameter count efficient.
        # e.g., if out_channels=64, each branch outputs 16 channels.
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 Conv (Pixel-wise transformation, similar to ResNet bottleneck)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Conv, Dilation=1 (Standard local context)
        # Padding=1 ensures spatial resolution is maintained.
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3 Conv, Dilation=2 (Medium context)
        # Padding=2 ensures spatial resolution is maintained.
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 Conv, Dilation=4 (Large context)
        # Crucial for fixing wide halos or "foggy" boundaries.
        # Padding=4 ensures spatial resolution is maintained.
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion Layer: Compresses the concatenated features back to 'out_channels'
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Parallel Processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 2. Concatenation (Channel-wise)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # 3. Fusion
        out = self.fusion(out)
        
        # 4. Residual Connection
        # Input 'x' is added to the output, allowing the network to learn 
        # only the "difference" (residual) needed to improve the features.
        out += x
        
        return self.relu(out)

class Refiner(nn.Module):
    """
    Residual Refinement Module (RRM) with Inception Architecture.
    
    Purpose:
        Takes the high-resolution RGB image and the 'Coarse Alpha' from the 
        TwinSwin backbone, and predicts a 'Residual Map' to correct errors.
        
    Mechanism:
        Final_Alpha = Sigmoid( Coarse_Alpha + Residual_Delta )
    """
    def __init__(self, in_channels=4, mid_channels=64):
        super().__init__()
        
        # 1. Feature Extraction (Entry)
        # Combines RGB (3) + Coarse Alpha (1) -> High dimensional feature space
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Refinement Body (Inception Blocks)
        # We stack 2 blocks to allow complex feature refinement.
        # Thanks to Dilation=4, the Receptive Field is large enough to see entire hair strands.
        self.body = nn.Sequential(
            DilatedInceptionBlock(mid_channels, mid_channels),
            DilatedInceptionBlock(mid_channels, mid_channels)
        )
        
        # 3. Residual Prediction Head
        # Collapses features to 1 channel (the correction map).
        # Note: No Activation (ReLU/Sigmoid) here. We need raw logits 
        # because the correction can be positive (add opacity) or negative (remove halo).
        self.conv_out = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, img, coarse_alpha):
        """
        Args:
            img: [B, 3, H, W] - Original RGB image.
            coarse_alpha: [B, 1, H, W] - Initial prediction from TwinSwin (0~1).
        
        Returns:
            refined_alpha: [B, 1, H, W] - Final alpha matte (0~1).
        """
        # Concatenate inputs along channel dimension
        x = torch.cat([img, coarse_alpha], dim=1)
        
        # Extract basic features
        x = self.conv_in(x)
        
        # Apply Multi-Scale Refinement
        x = self.body(x)
        
        # Predict the residual (Delta)
        residual = self.conv_out(x)
        
        # Apply Correction: Coarse + Delta
        # Sigmoid ensures the result stays strictly within [0, 1]
        refined_alpha = torch.sigmoid(coarse_alpha + residual)
        
        return refined_alpha