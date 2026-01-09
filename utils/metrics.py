# utils/metrics.py

import torch
import torch.nn.functional as F

def compute_gradient(img):
    """
    Compute image gradients (used for evaluating edge sharpness).
    Using Sobel filters to approximate gradients.
    """
    # Define Sobel kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(img.device)
    
    # Convolve with the kernels
    grad_x = F.conv2d(img, kernel_x, padding=1)
    grad_y = F.conv2d(img, kernel_y, padding=1)
    
    # Compute Gradient Magnitude
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

def calculate_matting_metrics(pred_alpha, gt_alpha):
    """
    Calculate Matting-specific metrics.
    
    Args:
        pred_alpha (Tensor): Predicted alpha matte (N, 1, H, W), range [0, 1]
        gt_alpha (Tensor): Ground truth alpha matte (N, 1, H, W), range [0, 1]
        
    Returns:
        mse (float): Mean Squared Error (Lower is better) - Good for checking convergence.
        sad (float): Sum of Absolute Differences (Lower is better) - The CORE metric for matting.
                     Returns k-SAD (divided by 1000).
        grad (float): Gradient Error (Lower is better) - Measures edge sharpness/quality.
                      Scaled by 1000 for easier reading.
        acc (float): Pixel-wise Accuracy (Higher is better) - Reference only (derived from MAD).
    """
    # Detach and ensure float type to save memory and avoid graph computation
    pred = pred_alpha.detach().float()
    gt = gt_alpha.detach().float()
    
    # 1. MSE (Mean Squared Error)
    # Standard loss metric, good for monitoring general convergence.
    mse = F.mse_loss(pred, gt).item()
    
    # 2. SAD (Sum of Absolute Differences)
    # The most important metric for Matting. 
    # Normalized by 1000 (k-SAD) for easier reading as raw values can be very large.
    sad = torch.abs(pred - gt).sum().item() / 1000.0
    
    # 3. Grad (Gradient Error)
    # Measures how well the model predicts fine details (hair strands, edges).
    pred_grad = compute_gradient(pred)
    gt_grad = compute_gradient(gt)
    # Multiplied by 1000 to make the number more readable.
    grad_err = F.mse_loss(pred_grad, gt_grad).item() * 1000 
    
    # 4. Accuracy (Pixel-wise Accuracy)
    # Derived from Mean Absolute Difference (MAD).
    # Note: Acc can be misleading in Matting due to large background areas.
    mad = torch.abs(pred - gt).mean().item()
    acc = (1.0 - mad) * 100.0
    
    return mse, sad, grad_err, acc