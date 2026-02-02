# utils/metrics.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_gradient(img):
    """
    Compute image gradients using Sobel filters.
    Used for evaluating edge sharpness (Gradient Error).
    
    Args:
        img (Tensor): Input image (B, 1, H, W)
    Returns:
        gradient_magnitude (Tensor): (B, 1, H, W)
    """
    # Define Sobel kernels
    # Created on the fly or cached to ensure device matching
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    # Perform convolution (padding=1 keeps same size)
    grad_x = F.conv2d(img, kernel_x, padding=1)
    grad_y = F.conv2d(img, kernel_y, padding=1)
    
    # Calculate gradient magnitude
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

@torch.no_grad()
def calculate_matting_metrics(pred_alpha, gt_alpha):
    """
    Calculate Matting-specific metrics.
    Note: Inputs are detached automatically due to @torch.no_grad()
    
    Args:
        pred_alpha (Tensor): Predicted alpha matte (B, 1, H, W), range [0, 1]
        gt_alpha (Tensor): Ground truth alpha matte (B, 1, H, W), range [0, 1]
        
    Returns:
        mse (float): Mean Squared Error
        sad (float): Mean Sum of Absolute Differences (Scaled by 1/1000)
        grad (float): Gradient Error (Scaled by 1000)
        acc (float): Pixel-wise Accuracy (%)
    """
    # Ensure inputs are float32
    pred = pred_alpha.float()
    gt = gt_alpha.float()
    
    batch_size = pred.size(0)

    # 1. MSE (Mean Squared Error)
    # F.mse_loss uses reduction='mean' by default (averages over batch & pixels), which is correct.
    mse = F.mse_loss(pred, gt).item()
    
    # 2. SAD (Sum of Absolute Differences)
    # [Correction] We must normalize by batch_size to make it comparable across Train/Val.
    # Calculate L1 sum per image -> Average over batch
    # Result is "Average SAD per image"
    l1_sum = torch.abs(pred - gt).view(batch_size, -1).sum(dim=1).mean()
    sad = l1_sum.item() / 1000.0  # Scaled by 1000 as requested (k-SAD)
    
    # 3. Grad (Gradient Error)
    # Evaluates edge sharpness.
    # Scaled by 1000 for easier reading.
    pred_grad = compute_gradient(pred)
    gt_grad = compute_gradient(gt)
    grad_err = F.mse_loss(pred_grad, gt_grad).item() * 1000 
    
    # 4. Accuracy (Pixel-wise Accuracy)
    # Derived from Mean Absolute Difference (MAD).
    # MAD is essentially L1 Loss (mean reduction)
    mad = F.l1_loss(pred, gt).item()
    acc = (1.0 - mad) * 100.0
    
    return mse, sad, grad_err, acc