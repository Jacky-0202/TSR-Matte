# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Standard U-Net Blocks
# ==========================================

class DoubleConv(nn.Module):
    """
    (Conv2d => BN => ReLU) * 2
    Standard U-Net building block.
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
    Upscaling block that handles skip connections.
    Includes padding logic to handle odd-sized feature maps (asymmetric shapes).
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling to reduce channel count before concatenation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The input to DoubleConv will be (in_channels//2 + skip_channels)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Current feature map (needs upsampling)
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # [Padding Logic]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==========================================
# 2. RSU Blocks (For Refiner / U^2-Net)
# ==========================================

class REBNCONV(nn.Module):
    """
    Basic convolution block for RSU.
    """
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate, bias=False)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

class RSU4(nn.Module):
    """
    Residual U-block (Depth=4).
    Corrected Decoder logic to handle upsampling before concatenation.
    """
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        # --- Encoder Part ---
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        # --- Bridge Part (Dilated) ---
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        # --- Decoder Part ---
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        # 1. Input Feature Extraction
        hx = x
        hxin = self.rebnconvin(hx)

        # 2. Encoder Path (Downsampling)
        hx1 = self.rebnconv1(hxin)
        hx = F.max_pool2d(hx1, 2, stride=2, ceil_mode=True)

        hx2 = self.rebnconv2(hx)
        hx = F.max_pool2d(hx2, 2, stride=2, ceil_mode=True)

        hx3 = self.rebnconv3(hx)
        hx = F.max_pool2d(hx3, 2, stride=2, ceil_mode=True)

        # 3. Bridge
        hx4 = self.rebnconv4(hx)

        # 4. Decoder Path (Upsampling -> Concatenation -> Conv)
        
        # Upsample hx4 first, then concat with hx3
        hx4up = self._upsample_like(hx4, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4up, hx3), 1))

        # Upsample hx3d first, then concat with hx2
        hx3dup = self._upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))

        # Upsample hx2d first, then concat with hx1
        hx2dup = self._upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        # 5. Residual Connection
        return hx1d + hxin

    def _upsample_like(self, src, tar):
        """
        Helper to upsample 'src' to match the size of 'tar'.
        """
        return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)