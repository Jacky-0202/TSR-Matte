# models/tsr_matte.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.blocks import DoubleConv, OutConv, BasicRFB, GradientModule, DilatedRefiner

def load_pretrained_1ch(model, backbone_name):
    """
    Helper function to load ImageNet pretrained weights for a 1-channel backbone.
    It averages the weights of the first convolution layer (RGB -> Grayscale).
    """
    try:
        temp_model = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        state_dict = temp_model.state_dict()
        
        if 'patch_embed.proj.weight' in state_dict:
            weight_3ch = state_dict['patch_embed.proj.weight']
            # Average the 3 RGB channels into 1 channel
            weight_1ch = weight_3ch.mean(dim=1, keepdim=True) 
            state_dict['patch_embed.proj.weight'] = weight_1ch
            
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ [gt_encoder] Initialized 1-channel backbone ({backbone_name}) with averaged ImageNet weights.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load pretrained weights for gt_encoder: {e}")

class TSRMatteNet(nn.Module):
    def __init__(self, n_classes=1, img_size=224, backbone_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        TSR-Matte: Twin Swin Refiner Matting Network.
        Architecture: LM (Swin) + Refiner (Dilated Inception) -> Residual Addition
        """
        super(TSRMatteNet, self).__init__()
        
        # ==========================================
        # 1. LM Module (TwinSwin Backbone)
        # ==========================================
        print(f"🏗️ [LM-Student] Backbone: {backbone_name}")
        self.img_encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, 
            out_indices=(0, 1, 2, 3), img_size=img_size, strict_img_size=False
        )
        
        print(f"🏗️ [LM-Teacher] Backbone: {backbone_name} (1-ch)")
        self.gt_encoder = timm.create_model(
            backbone_name, pretrained=False, in_chans=1, features_only=True, 
            out_indices=(0, 1, 2), img_size=img_size, strict_img_size=False
        )
        if pretrained: load_pretrained_1ch(self.gt_encoder, backbone_name)
        
        # Freeze Teacher
        for param in self.gt_encoder.parameters(): param.requires_grad = False

        # --- Decoder Settings ---
        if 'tiny' in backbone_name or 'small' in backbone_name: embed_dim = 96
        elif 'base' in backbone_name: embed_dim = 128
        elif 'large' in backbone_name: embed_dim = 192
        else: embed_dim = 96
        
        self.dims = [embed_dim * (2 ** i) for i in range(4)]
        DECODER_DIM = 64

        # Adapters
        self.adapt_c3 = nn.Sequential(nn.Conv2d(self.dims[2], self.dims[2], 1), nn.GroupNorm(32, self.dims[2]), nn.ReLU(True))

        # LM Decoder
        self.neck = BasicRFB(self.dims[2], DECODER_DIM)
        self.c4_project = nn.Sequential(nn.Conv2d(self.dims[3], DECODER_DIM, 1), nn.GroupNorm(16, DECODER_DIM), nn.ReLU(True))
        self.fusion_deep = DoubleConv(DECODER_DIM*2, DECODER_DIM)
        self.low_level_project = nn.Sequential(nn.Conv2d(self.dims[0], DECODER_DIM, 1), nn.GroupNorm(16, DECODER_DIM), nn.ReLU(True))
        self.decoder_head = DoubleConv(DECODER_DIM*2, DECODER_DIM)
        self.outc = OutConv(DECODER_DIM, n_classes) # LM Output (Coarse Mask)

        # ==========================================
        # 2. Refiner Module (Residual Predictor)
        # ==========================================
        print("✨ Building Dilated Refiner (Output: 1-ch Residual)")
        
        # Gradient Module
        self.grad_layer = GradientModule()
        
        # Refiner (Input: 4 channels, Output: 1 channel)
        self.refiner = DilatedRefiner(in_channels=4)
        self.res_scale = nn.Parameter(torch.tensor(0.15))

        # ==========================================
        # 3. No Fusion Layer Needed (Direct Add)
        # ==========================================

    def forward(self, x, gt_mask=None):
        input_shape = x.shape[-2:]
        
        # ---------------------------
        # Step 1: LM Module (Coarse Prediction)
        # ---------------------------
        img_raw = self.img_encoder(x)
        img_feats = [f.permute(0, 3, 1, 2).contiguous() if f.ndim==4 and f.shape[1]!=f.shape[-1] else f for f in img_raw]
        
        c1, c3, c4 = img_feats[0], img_feats[2], img_feats[3]
        c3_adapt = self.adapt_c3(c3)

        # Teacher Logic (Training Only)
        gt_encoder_c3 = None
        if self.training and gt_mask is not None:
            with torch.no_grad():
                gt_in = gt_mask.unsqueeze(1) if gt_mask.dim()==3 else gt_mask
                gt_raw = self.gt_encoder(gt_in.float())
                gt_feats = [f.permute(0, 3, 1, 2).contiguous() if f.ndim==4 and f.shape[1]!=f.shape[-1] else f for f in gt_raw]
                gt_encoder_c3 = gt_feats[2]

        # LM Decoder
        x_rfb = self.neck(c3)
        x_c4 = self.c4_project(c4)
        x_c4_up = F.interpolate(x_c4, size=x_rfb.shape[2:], mode='bilinear', align_corners=True)
        x_deep = self.fusion_deep(torch.cat([x_rfb, x_c4_up], dim=1))
        
        x_mid = F.interpolate(x_deep, size=c1.shape[2:], mode='bilinear', align_corners=True)
        c1_low = self.low_level_project(c1)
        x_dec = self.decoder_head(torch.cat([x_mid, c1_low], dim=1))
        
        # Get Coarse Mask (0~1)
        lm_logits = self.outc(x_dec)
        coarse_mask = torch.sigmoid(F.interpolate(lm_logits, size=input_shape, mode='bilinear', align_corners=True))

        # ---------------------------
        # Step 2: Refiner (Residual Prediction)
        # ---------------------------
        # 1. Gradient Map (Prior for edges)
        grad_map = self.grad_layer(x) # (B, 1, H, W)
        
        # 2. Refiner Input (RGB + Gradient) - Detached from LM
        refiner_input = torch.cat([x, grad_map], dim=1) # (B, 4, H, W)
        
        # 3. Predict Residual
        # Range: Unbounded (can be positive or negative)
        residual = self.refiner(refiner_input) # (B, 1, H, W)
        
        # 4 * x * (1 - x)
        # x=0 -> 0; x=1 -> 0; x=0.5 -> 1.0
        edge_attention = 4 * coarse_mask * (1 - coarse_mask)
        residual_masked = residual * edge_attention

        # ---------------------------
        # Step 3: Additive Fusion
        # ---------------------------
        # Final = Coarse + (w * Residual)
        final_alpha = coarse_mask + (self.res_scale * residual_masked)
        final_alpha = torch.clamp(final_alpha, 0, 1)

        return final_alpha, (c3_adapt,), (gt_encoder_c3,)