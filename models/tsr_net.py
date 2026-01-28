# models/tsr_net.py

import torch
import torch.nn as nn
import timm
from models.blocks import Up, DoubleConv, RSU4

class TSRNet(nn.Module):
    def __init__(self, n_classes=1, img_size=1024, backbone_name='swin_base_patch4_window7_224', pretrained=True):
        """
        TSRNet (Twin Swin Refiner Network):
        A high-precision matting architecture combining:
        1. Swin Transformer Encoder (Student)
        2. U-Net Decoder (Coarse Prediction)
        3. RSU Refiner (Fine Detail Refinement)
        """
        super(TSRNet, self).__init__()
        
        print(f"🏗️ Initializing TSRNet with backbone: {backbone_name}")

        # --- 1. Encoder (Backbone) ---
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size
        )

        self.dims = self.backbone.feature_info.channels()
        print(f"   Feature Channels: {self.dims}")
        c0, c1, c2, c3 = self.dims

        # --- 2. Decoder (Coarse Path) ---
        self.up1 = Up(c3, c2, c2)
        self.up2 = Up(c2, c1, c1)
        self.up3 = Up(c1, c0, c0)

        # Expand blocks to reach full resolution
        self.expand_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(c0, c0 // 2)
        )
        self.expand_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c0 // 2, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Coarse Head: Predicts the initial mask
        self.head = nn.Conv2d(24, n_classes, kernel_size=1)

        # --- 3. Refiner (Fine Path) ---
        # RSU-4 Module input: RGB Image (3) + Coarse Logits (1) = 4 Channels
        print("   Initializing RSU Refiner (RSU-4)...")
        self.refiner = RSU4(in_ch=4, mid_ch=32, out_ch=16)
        
        # Refiner Head: Predicts the residual difference
        self.refiner_out = nn.Conv2d(16, n_classes, kernel_size=3, padding=1)

    def _to_nchw(self, x, expected_c):
        """ Robustly converts (N, H, W, C) to (N, C, H, W). """
        if x.shape[1] == expected_c:
            return x
        elif x.shape[-1] == expected_c:
            return x.permute(0, 3, 1, 2).contiguous()
        else:
            return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        """
        Returns:
            If training: (refined_logits, coarse_logits, [features])
            If eval: refined_logits
        """
        # --- A. Encoder ---
        features_raw = self.backbone(x)
        features = []
        for i, f in enumerate(features_raw):
            features.append(self._to_nchw(f, self.dims[i]))
        x0, x1, x2, x3 = features

        # --- B. Decoder (Coarse) ---
        d1 = self.up1(x3, x2)
        d2 = self.up2(d1, x1)
        d3 = self.up3(d2, x0)
        d4 = self.expand_1(d3)
        d_final = self.expand_2(d4)
        
        # Coarse Prediction
        coarse_logits = self.head(d_final)
        
        # --- C. Refiner (Fine) ---
        # 1. Concatenate RGB Input (x) + Coarse Logits
        ref_input = torch.cat([x, coarse_logits], dim=1) # (B, 4, H, W)
        
        # 2. RSU Forward
        ref_feat = self.refiner(ref_input)
        
        # 3. Residual & Final Addition
        residual = self.refiner_out(ref_feat)
        refined_logits = coarse_logits + residual
        
        # --- D. Return Logic ---
        if self.training:
            # Return tuple for Deep Supervision & Feature Alignment
            return refined_logits, coarse_logits, [x2, x3]
        else:
            # Inference mode: return only the best result
            return refined_logits