# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    The standard building block of U-Net.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            # First Conv
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
            
            # Second Conv
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv.
    Updated to handle asymmetric channel sizes from Swin Transformer.
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Current feature map from decoder (needs upsampling)
        x2: Skip connection feature map from encoder
        """
        x1 = self.up(x1)
        
        # Handling Shape Mismatch (CHW)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis (dim=1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution to map features to classes.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class BasicRFB(nn.Module):
    """
    Receptive Field Block (RFB).
    Enhances feature discrimination and robustness using multi-scale dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        
        # Reduce intermediate channels for efficiency
        inter_channels = in_channels // 8

        # --- Branch 0: 1x1 Conv ---
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Branch 1: 3x3 Conv (Dilation Rate 1) ---
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Branch 2: 3x3 Conv (Dilation Rate 3) ---
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=3, dilation=3, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Branch 3: 3x3 Conv (Dilation Rate 5) ---
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=5, dilation=5, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Feature Fusion ---
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inter_channels * 4, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels), 
            nn.ReLU(inplace=True)
        )
        
        # --- Shortcut Connection ---
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)
        
        x_out = self.shortcut(x) + x_cat * self.scale
        return F.relu(x_out, inplace=True)
    

# --- [Gradient Module (Fixed for Noise Reduction)] ---
class GradientModule(nn.Module):
    def __init__(self):
        super(GradientModule, self).__init__()
        # Define Sobel Filters (Fixed, no training needed)
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        self.register_buffer('gray_weight', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x):
        # Convert to Grayscale
        gray = F.conv2d(x, self.gray_weight)
        
        # Calculate Gradients
        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        
        # Gradient Magnitude
        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # [Instance-wise Normalization]
        B, C, H, W = gradient.shape
        g_max = gradient.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        gradient = gradient / (g_max + 1e-8)
        
        # Filter out weak gradients (background noise/texture).
        # Any edge weaker than 0.1 is suppressed to 0.
        gradient = F.relu(gradient - 0.1)
        
        return gradient

# --- [Dilated Refiner] ---
class DilatedRefiner(nn.Module):
    """
    Refiner Module specialized for Residual Prediction.
    Input: RGB(3) + Gradient(1) = 4 channels
    Output: Residual Map (1 channel)
    
    Key Features:
    1. Shallow & High-Res (No Downsampling).
    2. No ReLU in the final layer (Allows negative residuals).
    3. Zero Initialization (Ensures stable cold start).
    """
    def __init__(self, in_channels=4):
        super(DilatedRefiner, self).__init__()
        
        hidden_dim = 32
        
        # Layer 1: Basic Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Multi-scale Dilated Convs (Inception Style)
        self.branch1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=4, dilation=4, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # --- Output Layer ---
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        
        # [Zero Initialization]
        nn.init.constant_(self.final_conv.weight, 0)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        
        b1 = self.branch1(x1)
        b2 = self.branch2(x1)
        b3 = self.branch3(x1)
        b4 = self.branch4(x1)
        
        # Concatenate branches
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Predict Residual
        residual = self.final_conv(concat)
        
        return residual