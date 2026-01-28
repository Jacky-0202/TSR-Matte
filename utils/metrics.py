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
    # Define Sobel kernels (fixed, no gradients needed)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    # Perform convolution
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
        pred_alpha (Tensor): Predicted alpha matte (N, 1, H, W), range [0, 1]
        gt_alpha (Tensor): Ground truth alpha matte (N, 1, H, W), range [0, 1]
        
    Returns:
        mse (float): Mean Squared Error
        sad (float): Sum of Absolute Differences (Scaled by 1/1000 => k-SAD)
        grad (float): Gradient Error (Scaled by 1000 for readability)
        acc (float): Pixel-wise Accuracy (%)
    """
    # Ensure inputs are float32
    pred = pred_alpha.float()
    gt = gt_alpha.float()
    
    # 1. MSE (Mean Squared Error)
    # Standard metric for convergence monitoring.
    mse = F.mse_loss(pred, gt).item()
    
    # 2. SAD (Sum of Absolute Differences)
    # The most critical metric for Matting tasks.
    # Returns k-SAD (divided by 1000) for easier reading.
    sad = torch.abs(pred - gt).sum().item() / 1000.0
    
    # 3. Grad (Gradient Error)
    # Evaluates edge sharpness and fine details.
    # Scaled by 1000 since raw values are typically very small.
    pred_grad = compute_gradient(pred)
    gt_grad = compute_gradient(gt)
    grad_err = F.mse_loss(pred_grad, gt_grad).item() * 1000 
    
    # 4. Accuracy (Pixel-wise Accuracy)
    # Derived from Mean Absolute Difference (MAD).
    mad = torch.abs(pred - gt).mean().item()
    acc = (1.0 - mad) * 100.0
    
    return mse, sad, grad_err, acc