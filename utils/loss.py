import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ==========================================
# 1. Structure Loss (Derived from BiRefNet / F3Net)
#    Core Function: Edge-weighted perception.
#    It assigns higher weights to boundary areas to fix inaccurate edges.
# ==========================================
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        """
        Args:
            pred: Model Logits (B, 1, H, W) -> Before Sigmoid
            mask: Ground Truth (B, 1, H, W) -> 0 or 1
        """
        # 1. Generate Edge-Weighted Map (weit)
        # Apply AvgPool to GT to blur it, then subtract from original GT.
        # The absolute difference is maximized at the edges.
        wb = 1.0
        target = mask.float()
        
        # Uses a 31x31 Kernel to detect edges
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        
        # 2. Weighted BCE Loss (Focuses heavily on edges)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # 3. Weighted IoU Loss (Structure-aware)
        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        # Return the average of both Weighted BCE and Weighted IoU
        return (wbce + wiou).mean()

# ==========================================
# 2. Gradient Loss (Essential for Matting)
#    Core Function: Enforces gradient consistency.
#    Ensures that the predicted edges are as sharp as the GT.
# ==========================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel Kernels for X and Y directions
        kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.kernel_x = kernel_x.view((1, 1, 3, 3))
        self.kernel_y = kernel_y.view((1, 1, 3, 3))

    def forward(self, pred, gt):
        # pred, gt: (B, 1, H, W) Probability maps [0, 1]
        
        # Ensure kernels are on the same device as input
        if pred.device != self.kernel_x.device:
            self.kernel_x = self.kernel_x.to(pred.device)
            self.kernel_y = self.kernel_y.to(pred.device)

        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)

        # Calculate L1 distance between gradients (Enforces sharpness)
        grad_loss = torch.abs(pred_grad_x - gt_grad_x) + torch.abs(pred_grad_y - gt_grad_y)
        return grad_loss.mean()

# ==========================================
# 3. SSIM Loss (Auxiliary Structural Integrity)
#    Core Function: Maintains local structural similarity (luminance/contrast/structure).
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
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

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        # Check if window needs to be re-initialized (e.g., if device changed)
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
            
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# ==========================================
# 4. Main Loss Wrapper (Deep Supervision Ready)
#    Combines BiRefNet's Structure Loss with Matting Losses.
#    Supports Tuple inputs for both Refined and Coarse predictions.
# ==========================================
class MattingLoss(nn.Module):
    def __init__(self, 
                 weight_struct=1.0, # [BiRefNet] Structure & Edge weighting (Includes weighted BCE + IoU)
                 weight_l1=1.0,     # [Matting] Pixel-level absolute error (Core for regression)
                 weight_grad=1.0,   # [Matting] Gradient sharpness
                 weight_ssim=0.5,   # [Common] Structural integrity
                 weight_feat=0.2):  # [Twin] Feature alignment (if Teacher is used)
        super(MattingLoss, self).__init__()
        
        self.weight_struct = weight_struct
        self.weight_l1 = weight_l1
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_feat = weight_feat
        
        # Initialize Loss Modules
        self.structure_loss = StructureLoss()
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()
        self.mse = nn.MSELoss()

    def calculate_single_loss(self, pred_logits, gt_f32):
        """ 
        Helper function to calculate the weighted sum of losses for a single prediction map.
        """
        # 1. Structure Loss (Uses Raw Logits)
        loss_struct = 0.0
        if self.weight_struct > 0:
            loss_struct = self.structure_loss(pred_logits, gt_f32)

        # 2. Generate Probability Map (Sigmoid)
        pred_prob = torch.sigmoid(pred_logits)
        
        # 3. Calculate Pixel-level Losses
        loss_l1 = self.l1(pred_prob, gt_f32) if self.weight_l1 > 0 else 0.0
        loss_grad = self.grad(pred_prob, gt_f32) if self.weight_grad > 0 else 0.0
        loss_ssim = self.ssim(pred_prob, gt_f32) if self.weight_ssim > 0 else 0.0
        
        # 4. Summation
        total = (self.weight_struct * loss_struct) + \
                (self.weight_l1 * loss_l1) + \
                (self.weight_grad * loss_grad) + \
                (self.weight_ssim * loss_ssim)
                
        return total, loss_l1

    def forward(self, preds, gt_alpha, stu_feats=None, tea_feats=None):
        """
        Args:
            preds: Can be a single Tensor OR a Tuple (refined_logits, coarse_logits)
            gt_alpha: Ground Truth (B, 1, H, W)
            stu_feats: Student Features
            tea_feats: Teacher Features
        """
        gt_f32 = gt_alpha.float()
        
        # --- 1. Content Loss (Deep Supervision) ---
        total_content_loss = 0.0
        main_l1 = 0.0 # Used for logging the L1 of the final output
        
        if isinstance(preds, (tuple, list)):
            # [Refiner Mode]
            # preds[0] = Refined (Final) -> Weight 1.0 (Primary Focus)
            # preds[1] = Coarse (Backbone) -> Weight 0.5 (Auxiliary Supervision)
            
            # A. Refined Loss (Priority: High)
            loss_refined, l1_refined = self.calculate_single_loss(preds[0], gt_f32)
            total_content_loss += loss_refined
            main_l1 = l1_refined
            
            # B. Coarse Loss (Priority: Medium - prevents backbone degradation)
            if len(preds) > 1:
                loss_coarse, _ = self.calculate_single_loss(preds[1], gt_f32)
                total_content_loss += 0.5 * loss_coarse
                
        else:
            # [Standard Mode] Single output
            total_content_loss, main_l1 = self.calculate_single_loss(preds, gt_f32)

        # --- 2. Feature Alignment Loss (Twin Strategy) ---
        loss_feat = torch.tensor(0.0, device=gt_alpha.device)
        if self.weight_feat > 0 and stu_feats is not None and tea_feats is not None:
            if isinstance(stu_feats, (list, tuple)):
                for s_f, t_f in zip(stu_feats, tea_feats):
                    loss_feat += self.mse(s_f.float(), t_f.float())
            else:
                loss_feat += self.mse(stu_feats.float(), tea_feats.float())
        
        # --- 3. Final Summation ---
        total_loss = total_content_loss + (self.weight_feat * loss_feat)
        
        # Return components for logging (Loss, L1, Content, Feature)
        return total_loss, main_l1, total_content_loss, loss_feat