# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from config import Config

# ==========================================
# 1. Structure Loss
#    Combines Weighted BCE and Weighted IoU.
#    Focuses on the overall shape and structural integrity.
#    Based on the logic often used in SOD/COD tasks (e.g., BiRefNet).
# ==========================================
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        """
        Args:
            pred: Model Logits (B, 1, H, W) -> Before Sigmoid
            mask: Ground Truth (B, 1, H, W) -> 0~1 Float
        """
        # 1. Generate Edge-Weighted Map (weit)
        # This weight map emphasizes boundaries where the ground truth has high variance.
        wb = 1.0
        target = mask.float()
        
        # Use AvgPool to find edges (areas with high local variance)
        # padding=15 matches kernel_size=31 to keep size consistent
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        
        # 2. Weighted BCE Loss
        # Clamp logits for numerical stability to prevent NaN
        pred = torch.clamp(pred, min=-10, max=10) 
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # Apply the weight map to focus BCE on hard examples (edges)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # 3. Weighted IoU Loss
        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        # Return average of Weighted BCE + Weighted IoU
        return (wbce + wiou).mean()


# ==========================================
# 2. Gradient Loss (Sharpness Enforcer)
#    Penalizes differences in image gradients (edges) between prediction and GT.
#    Crucial for reducing "halos" and ensuring sharp alpha mattes.
# ==========================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel Kernels for X and Y directions
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Register kernels as buffers so they are saved with the model but not updated by optimizer
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, pred, gt):
        # pred should be probabilities (0~1), not logits
        
        # Compute gradients for Prediction
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        
        # Compute gradients for Ground Truth
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)

        # L1 distance between the gradient maps
        loss = torch.abs(pred_grad_x - gt_grad_x) + torch.abs(pred_grad_y - gt_grad_y)
        return loss.mean()


# ==========================================
# 3. SSIM Loss (Structural Consistency)
#    Ensures the luminance, contrast, and structure match the ground truth.
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        # Generate 1D Gaussian kernel
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        # Create 2D Gaussian window
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # Calculate local statistics (mean, variance, covariance)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        # SSIM formula
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # Dynamically move window to the correct device/type if needed
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
            
        # Return 1 - SSIM (since we want to minimize loss)
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


# ==========================================
# 4. Total Loss Factory (The Adapter)
#    Updated for Residual Refinement Module (RRM) Strategy.
#    Handles Deep Supervision for both Coarse (Student) and Refined (RRM) outputs.
# ==========================================
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        
        # Load weights from Config (The Control Center)
        self.weights = Config.LOSS_WEIGHTS
        
        # Initialize components
        self.structure_loss = StructureLoss() # Combined Weighted BCE + IoU
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()
        self.mse = nn.MSELoss() # For Feature Alignment

    def compute_single_loss(self, pred_logits, target, use_grad_boost=False):
        """
        Helper function to compute loss for a single output map.
        
        Args:
            pred_logits: Raw model output (before Sigmoid).
            target: Ground truth mask.
            use_grad_boost: If True, doubles the Gradient Loss weight (useful for Refiner).
            
        Returns:
            loss_val: The scalar loss value.
            log_dict: A dictionary of loss components for logging.
        """
        loss_val = 0.0
        log_dict = {}
        
        # 1. Structure Loss (Weighted BCE + IoU)
        # Often used for Locator mode or providing base structural supervision.
        # We combine 'bce' and 'dice' weights from config for this.
        w_struct = self.weights.get('bce', 0.0) + self.weights.get('dice', 0.0)
        if w_struct > 0:
            l_struct = self.structure_loss(pred_logits, target)
            loss_val += w_struct * l_struct
            log_dict['struct'] = l_struct.item()

        # Compute probabilities for other losses
        pred_prob = torch.sigmoid(pred_logits)

        # 2. L1 Loss (Pixel-level accuracy)
        w_l1 = self.weights.get('l1', 0.0)
        if w_l1 > 0:
            l_l1 = self.l1(pred_prob, target)
            loss_val += w_l1 * l_l1
            log_dict['l1'] = l_l1.item()

        # 3. Gradient Loss (Edge Sharpness)
        w_grad = self.weights.get('grad', 0.0)
        
        # [RRM Strategy]
        # If this is the Refiner stage, we aggressively boost gradient supervision
        # to force the network to eliminate halos and sharpen edges.
        if use_grad_boost:
            # Ensure weight is at least 1.0, or double the config value
            w_grad = max(w_grad * 2.0, 1.0) 
            
        if w_grad > 0:
            l_grad = self.grad(pred_prob, target)
            loss_val += w_grad * l_grad
            log_dict['grad'] = l_grad.item()

        # 4. SSIM Loss (Structural Consistency)
        w_ssim = self.weights.get('ssim', 0.0)
        if w_ssim > 0:
            l_ssim = self.ssim(pred_prob, target)
            loss_val += w_ssim * l_ssim
            log_dict['ssim'] = l_ssim.item()
            
        return loss_val, log_dict

    def forward(self, outputs, target, teacher_feats=None):
        """
        Unified Forward Pass.
        
        Args:
            outputs (dict): 
                - 'coarse' (or 'fine'): Logits from TwinSwin (Student).
                - 'refined': Logits from Refiner (Optional, if RRM is used).
                - 'feats': Student Features for alignment.
            target (Tensor): GT Mask (B, 1, H, W).
            teacher_feats (list): Teacher Features (Optional, passed from train.py).
        
        Returns:
            total_loss (Tensor): Scalar for backward.
            loss_dict (dict): Breakdown for logging.
        """
        # Ensure target is float
        target = target.float()
        
        total_loss = 0.0
        loss_dict_agg = {}

        # --- A. Coarse Stage (TwinSwin Output) ---
        # The Student network's output. We call it 'coarse' in the RRM context,
        # but it might be named 'fine' in the original code. We check both.
        if 'coarse' in outputs:
            coarse_logits = outputs['coarse']
        else:
            # Fallback to 'fine' if 'coarse' key is missing
            coarse_logits = outputs['fine']
            
        # Compute loss for Coarse stage (No Gradient Boost)
        # We allow some uncertainty/halos here to give the Refiner something to work with.
        l_coarse, d_coarse = self.compute_single_loss(coarse_logits, target, use_grad_boost=False)
        total_loss += l_coarse
        
        # Log coarse losses with prefix
        for k, v in d_coarse.items():
            loss_dict_agg[f"coarse_{k}"] = v

        # --- B. Refined Stage (Refiner Output) ---
        # If the RRM is active, it produces a 'refined' output.
        # This is the final result, so we apply stricter supervision (Gradient Boost).
        if 'refined' in outputs:
            refined_logits = outputs['refined']
            l_refined, d_refined = self.compute_single_loss(refined_logits, target, use_grad_boost=True)
            total_loss += l_refined
            
            # Log refined losses with prefix
            for k, v in d_refined.items():
                loss_dict_agg[f"fine_{k}"] = v

        # --- C. Feature Alignment (Twin Strategy) ---
        # This only applies to the Student network features.
        stu_feats = outputs.get('feats', None)
        w_feat = self.weights.get('feat', 0.0)
        
        if w_feat > 0 and stu_feats is not None and teacher_feats is not None:
            l_feat = 0.0
            # Align features layer by layer (e.g., H/16, H/8)
            min_len = min(len(stu_feats), len(teacher_feats))
            for i in range(min_len):
                # Using MSE for feature matching
                l_feat += self.mse(stu_feats[i].float(), teacher_feats[i].float())
            
            total_loss += w_feat * l_feat
            loss_dict_agg['feat'] = l_feat.item()

        return total_loss, loss_dict_agg