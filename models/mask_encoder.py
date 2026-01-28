# models/mask_encoder.py
import torch
import torch.nn as nn
from models.blocks import DoubleConv

class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports (N, C, H, W) layout directly.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PatchMerging(nn.Module):
    """
    Simulates Swin Transformer's Patch Merging (Downsampling).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.norm(x)
        return x

class SwinMaskEncoder(nn.Module):
    """
    Lightweight Teacher Network to extract features from GT Masks.
    Now supports dynamic embedding dimensions (e.g., 96 for Tiny, 128 for Base).
    """
    def __init__(self, embed_dim=96):
        super().__init__()
        # Determine channels based on embed_dim
        # Tiny: 96 -> [96, 192, 384, 768]
        # Base: 128 -> [128, 256, 512, 1024]
        
        self.c0 = embed_dim
        self.c1 = embed_dim * 2
        self.c2 = embed_dim * 4
        self.c3 = embed_dim * 8
        
        print(f"🎓 MaskEncoder initialized with dims: [{self.c0}, {self.c1}, {self.c2}, {self.c3}]")

        # Stage 0: 4x Downsample (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.c0, kernel_size=4, stride=4),
            LayerNorm2d(self.c0)
        )
        
        # Stage 1: c0 -> c1 (H/8)
        self.stage1 = PatchMerging(self.c0, self.c1)
        
        # Stage 2: c1 -> c2 (H/16) - Alignment Target 1
        self.stage2 = PatchMerging(self.c1, self.c2)
        self.block2 = DoubleConv(self.c2, self.c2) 
        
        # Stage 3: c2 -> c3 (H/32) - Alignment Target 2
        self.stage3 = PatchMerging(self.c2, self.c3)
        self.block3 = DoubleConv(self.c3, self.c3)

    def forward(self, x):
        # x: (B, 1, 1024, 1024)
        x = self.stem(x)   # (B, c0, 256, 256)
        x = self.stage1(x) # (B, c1, 128, 128)
        
        x = self.stage2(x) # (B, c2, 64, 64)
        f2 = self.block2(x)
        
        x = self.stage3(f2) # (B, c3, 32, 32)
        f3 = self.block3(x)
        
        # Return features corresponding to Stage 2 and Stage 3
        return [f2, f3]