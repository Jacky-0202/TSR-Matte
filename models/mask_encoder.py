# models/mask_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv

class MaskEncoder(nn.Module):
    """
    Teacher Network: Pure FPN architecture.
    Extracts high-quality features from masks to guide the Student.
    Aligned with Student output in both spatial resolution and channel count.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        # Expected dimensions for Swin-Base alignment: [128, 256, 512, 1024]
        self.dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]

        # --- 1. Bottom-up Pathway (Encoder) ---
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.dims[0], kernel_size=4, stride=4), 
            nn.GroupNorm(8, self.dims[0]), 
            nn.GELU()
        )
        # Downsampling stages to match the Student's hierarchy
        self.down1 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[1], kernel_size=2, stride=2), 
            DoubleConv(self.dims[1], self.dims[1])
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=2, stride=2), 
            DoubleConv(self.dims[2], self.dims[2])
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.dims[2], self.dims[3], kernel_size=2, stride=2), 
            DoubleConv(self.dims[3], self.dims[3])
        )

        # --- 2. Top-down Pathway (FPN Logic) ---
        # Lateral Connections: Align channels for feature fusion
        self.lat3_to_2 = nn.Conv2d(self.dims[3], self.dims[2], kernel_size=1) # 1024 -> 512
        self.lat2_to_1 = nn.Conv2d(self.dims[2], self.dims[1], kernel_size=1) # 512  -> 256

        # Smoothing Layers: Refine fused features and eliminate aliasing
        self.smooth2 = nn.Conv2d(self.dims[2], self.dims[2], kernel_size=3, padding=1) # H/16, 512ch
        self.smooth1 = nn.Conv2d(self.dims[1], self.dims[1], kernel_size=3, padding=1) # H/8,  256ch

    def forward(self, x):
        """
        Forward pass returning a list of features aligned with the Student's decoder.
        Output: [p2 (H/16, 512ch), p1 (H/8, 256ch)]
        """
        # --- 1. Bottom-up ---
        c0 = self.stem(x)   # H/4
        c1 = self.down1(c0) # H/8
        c2 = self.down2(c1) # H/16
        c3 = self.down3(c2) # H/32

        # --- 2. Top-down Fusion ---
        # Level 1: H/32 -> H/16 fusion
        p3_lat = self.lat3_to_2(c3)
        p3_up = F.interpolate(p3_lat, size=c2.shape[-2:], mode='nearest')
        p2 = self.smooth2(c2 + p3_up) # Refined feature at H/16

        # Level 2: H/16 -> H/8 fusion
        p2_lat = self.lat2_to_1(p2)
        p2_up = F.interpolate(p2_lat, size=c1.shape[-2:], mode='nearest')
        p1 = self.smooth1(c1 + p2_up) # Refined feature at H/8

        # --- 3. Pure Alignment Output ---
        return [p2, p1] # Matches Student's d1 and d2 perfectly