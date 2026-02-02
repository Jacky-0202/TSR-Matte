# models/refiner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic Residual Block with Dilation support.
    
    This block maintains the spatial resolution (no stride) and channel depth.
    It uses dilation to expand the receptive field, allowing the network to 
    observe 'context' (e.g., hair vs. background texture) around the edges 
    without reducing resolution.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 
            kernel_size=3, stride=1, 
            padding=dilation, dilation=dilation, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            channels, channels, 
            kernel_size=3, stride=1, 
            padding=1, dilation=1, # Second conv usually keeps dilation=1 to refine features
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection: The block learns the "difference" needed
        out += identity
        out = self.relu(out)
        
        return out

class Refiner(nn.Module):
    """
    Residual Refinement Module (RRM).
    
    Purpose:
        Takes the high-resolution RGB image and the 'Coarse Alpha' from the 
        TwinSwin backbone, and predicts a 'Residual Map' to correct errors 
        (mainly halos and fuzzy boundaries).
        
    Architecture:
        - Input: 4 Channels (3 RGB + 1 Coarse Alpha)
        - Body: A series of Residual Blocks with varying dilation rates.
        - Output: 1 Channel (Refined Alpha)
        - Operation: Final_Alpha = Sigmoid(Coarse_Alpha + Predicted_Residual)
    """
    def __init__(self, in_channels=4, mid_channels=64):
        super().__init__()
        
        # 1. Feature Extraction
        # Projects the 4-channel input into a higher-dimensional feature space.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Refinement Body (Bottleneck)
        # We use a sequence of blocks with different dilation rates to capture
        # multi-scale context information (useful for different hair thicknesses).
        self.body = nn.Sequential(
            BasicBlock(mid_channels, dilation=1),
            BasicBlock(mid_channels, dilation=2), # Look further for texture context
            BasicBlock(mid_channels, dilation=1)
        )
        
        # 3. Residual Prediction Head
        # Compresses features back to 1 channel (the correction map).
        # We do NOT use ReLU or Sigmoid here because the residual can be negative 
        # (to suppress halos) or positive (to fill holes).
        self.conv_out = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, img, coarse_alpha):
        """
        Args:
            img: [B, 3, H, W] - The original high-resolution RGB image.
            coarse_alpha: [B, 1, H, W] - The alpha prediction from TwinSwin (0~1 range).
        
        Returns:
            refined_alpha: [B, 1, H, W] - The final corrected alpha matte.
        """
        # Concatenate RGB and Coarse Alpha along the channel dimension
        # Input shape becomes [B, 4, H, W]
        x = torch.cat([img, coarse_alpha], dim=1)
        
        # Extract features
        x = self.conv_in(x)
        
        # Pass through residual blocks
        x = self.body(x)
        
        # Predict the residual (correction map)
        # This map contains values like +0.1 (add opacity) or -0.2 (remove halo)
        residual = self.conv_out(x)
        
        # Apply the correction: Coarse + Residual
        # Use Sigmoid to ensure the final output is strictly between 0 and 1
        refined_alpha = torch.sigmoid(coarse_alpha + residual)
        
        return refined_alpha