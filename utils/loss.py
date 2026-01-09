# utils/loss.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, gt):
        # Soft IoU calculation
        intersection = (pred * gt).sum(dim=(2, 3))
        total = (pred + gt).sum(dim=(2, 3))
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()

# --- [NEW] Gradient Loss ---
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel filters
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, gt):
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)
        
        # Gradient magnitude
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        gt_grad = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-8)
        
        # L1 loss on gradients
        return F.l1_loss(pred_grad, gt_grad)

class MattingLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_l1=1.0, weight_ssim=0.5, weight_iou=0.5, weight_grad=1.0, weight_feat=0.0):
        super(MattingLoss, self).__init__()
        
        self.weight_bce = weight_bce
        self.weight_l1 = weight_l1
        self.weight_ssim = weight_ssim
        self.weight_iou = weight_iou
        self.weight_grad = weight_grad # [NEW] Gradient Loss Weight
        self.weight_feat = weight_feat
        
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.iou = IoULoss()
        self.ssim = SSIMLoss(window_size=11)
        self.grad_loss = GradientLoss() # [NEW] Initialize Gradient Loss
        

    def forward(self, pred_alpha, gt_alpha, stu_feats=None, tea_feats=None):
        """
        Calculate composite matting loss.
        """
        epsilon = 1e-6
        
        pred_f32 = pred_alpha.float()
        gt_f32 = gt_alpha.float()

        pred_clamped = torch.clamp(pred_f32, epsilon, 1.0 - epsilon)
        gt_clamped = torch.clamp(gt_f32, 0.0, 1.0)
        
        # --- 1. BCE Loss ---
        with torch.autocast(device_type='cuda', enabled=False):
            loss_bce = self.bce(pred_clamped, gt_clamped)
        
        # --- 2. L1 Loss ---
        loss_l1 = self.l1(pred_clamped, gt_clamped)
        
        # --- 3. SSIM Loss ---
        loss_ssim = self.ssim(pred_clamped, gt_clamped)
        
        # --- 4. IoU Loss ---
        loss_iou = self.iou(pred_clamped, gt_clamped)

        # --- [NEW] 5. Gradient Loss ---
        loss_grad = self.grad_loss(pred_clamped, gt_clamped)
        
        # --- 6. Feature Consistency Loss ---
        loss_feat = 0.0
        if self.weight_feat > 0 and stu_feats is not None and tea_feats is not None:
            with torch.autocast(device_type='cuda', enabled=False):
                for s_f, t_f in zip(stu_feats, tea_feats):
                    loss_feat += self.mse(s_f.float(), t_f.float())
        
        # --- Total Loss ---
        total_loss = (self.weight_bce * loss_bce) + \
                     (self.weight_l1 * loss_l1) + \
                     (self.weight_ssim * loss_ssim) + \
                     (self.weight_iou * loss_iou) + \
                     (self.weight_grad * loss_grad) + \
                     (self.weight_feat * loss_feat)
        
        # Loss detail for monitoring
        loss_detail = loss_ssim + loss_iou + loss_grad
        
        return total_loss, loss_l1, loss_detail, loss_feat