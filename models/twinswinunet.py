# models/twinswinunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .blocks import DoubleConv, Up
from config import Config

class TwinSwinUNet(nn.Module):
    """
    Student Network: Swin Transformer Encoder with a CNN U-Net Decoder.
    Refactored for dimension consistency and cleaner forward logic.
    """
    def __init__(self):
        super().__init__()
        self.safe_size = Config.SAFE_SIZE
        
        # 1. Encoder (Swin Transformer via timm)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, 
            pretrained=True, 
            features_only=True,
            out_indices=(0, 1, 2, 3), 
            img_size=self.safe_size
        )
        # Channels for Swin-Base: [128, 256, 512, 1024]
        dims = self.backbone.feature_info.channels()

        # 2. Decoder (Standard CNN U-Net Blocks)
        # Up(in_channels, skip_channels, out_channels)
        self.up1 = Up(dims[3], dims[2], 512)  # H/32 + H/16 -> 512ch (H/16)
        self.up2 = Up(512, dims[1], 256)      # H/16 + H/8  -> 256ch (H/8)
        self.up3 = Up(256, dims[0], 128)      # H/8  + H/4  -> 128ch (H/4)
        
        # 3. Refined Output Head
        # Gradually recovers resolution from H/4 to H/1 with learnable layers
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H/4 -> H/2
            DoubleConv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # H/2 -> H/1
            DoubleConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def _fix_swin_format(self, features):
        """
        Internal utility to convert Swin's NHWC format back to CNN's NCHW format.
        This keeps the forward pass clean and readable.
        """
        fixed = []
        for i, f in enumerate(features):
            # Check if the last dimension matches the expected backbone channel count
            if f.dim() == 4 and f.shape[-1] == self.backbone.feature_info.channels()[i]:
                f = f.permute(0, 3, 1, 2).contiguous()
            fixed.append(f)
        return fixed

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Spatial Alignment: Resize input to the backbone's Safe Size (e.g., 896)
        x_inner = F.interpolate(x, size=(self.safe_size, self.safe_size), 
                               mode='bilinear', align_corners=False)
        
        # Encoder Forward
        feats = self.backbone(x_inner)
        x0, x1, x2, x3 = self._fix_swin_format(feats)

        # Decoder Forward
        d1 = self.up1(x3, x2) # Target Level 1: H/16, 512ch
        d2 = self.up2(d1, x1) # Target Level 2: H/8,  256ch
        d3 = self.up3(d2, x0) # Deepest decoder output: H/4, 128ch

        # Resolution Recovery
        logits_safe = self.head(d3)
        # Final interpolation to match the original input resolution (e.g., 1024)
        final_logits = F.interpolate(logits_safe, size=input_size, 
                                     mode='bilinear', align_corners=True)

        if self.training:
            # Return final prediction and features for alignment with Teacher
            return {'fine': final_logits, 'feats': [d1, d2]}
        return torch.sigmoid(final_logits)