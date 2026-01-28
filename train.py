# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import Custom Modules
import config
from utils.dataset import DIS5KDataset

# Import from the new TSR-Net file
from models.tsr_net import TSRNet 
from models.mask_encoder import SwinMaskEncoder
from utils.loss import MattingLoss
from utils.logger import CSVLogger
from utils.metrics import calculate_matting_metrics
from utils.plot import plot_history

def train():
    # --- 1. Setup ---
    print(f"🚀 Starting training: {config.EXPERIMENT_NAME}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Input Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"   Dilation: {config.DILATE_MASK} (Should be False for Refiner)")
    print(f"   Twin Alignment: {config.USE_TWIN_ALIGNMENT}")

    # Create directories
    logger = CSVLogger(config.LOG_DIR)
    
    # --- 2. Data Loading ---
    print("📂 Loading Datasets...")
    # [Matting Update] Dilation is OFF for high-precision training
    train_ds = DIS5KDataset(config.DATASET_ROOT, mode='train', 
                            target_size=config.IMG_SIZE, dilate_mask=config.DILATE_MASK)
    val_ds = DIS5KDataset(config.DATASET_ROOT, mode='val', 
                          target_size=config.IMG_SIZE, dilate_mask=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=config.NUM_WORKERS, 
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=1, 
                            shuffle=False, num_workers=config.NUM_WORKERS, 
                            pin_memory=config.PIN_MEMORY)

    print(f"   Train Images: {len(train_ds)}")
    print(f"   Val Images: {len(val_ds)}")

    # --- 3. Model Initialization ---
    print(f"🔹 Initializing TSR-Net (Student + Refiner): {config.BACKBONE_NAME}")
    model = TSRNet(n_classes=1, img_size=config.IMG_SIZE, 
                         backbone_name=config.BACKBONE_NAME).to(config.DEVICE)
    
    # B. Teacher Model
    teacher = None
    if config.USE_TWIN_ALIGNMENT:
        # Get embed_dim dynamically from the Student model
        student_embed_dim = model.dims[0]
        print(f"🎓 Initializing Teacher (Mask Encoder) with embed_dim={student_embed_dim}...")
        teacher = SwinMaskEncoder(embed_dim=student_embed_dim).to(config.DEVICE)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False 

    # --- 4. Optimization ---
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda') 
    
    # [Auto Config] This will load weights from config.LOSS_WEIGHTS
    loss_fn = MattingLoss(**config.LOSS_WEIGHTS).to(config.DEVICE)

    # --- 5. Training Loop ---
    best_iou = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_sad':[], 'val_sad':[], 
               'train_grad':[], 'val_grad':[], 'train_mse':[], 'val_mse':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # ==========================
        #       Training Phase
        # ==========================
        model.train()
        train_loss_epoch = 0
        train_metrics = [0, 0, 0, 0] # mse, sad, grad, acc

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]")
        
        for images, masks in train_loop:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            with torch.amp.autocast('cuda'):
                # 1. Teacher Forward
                tea_feats = None
                if teacher is not None:
                    with torch.no_grad():
                        tea_feats = teacher(masks)

                # 2. Student Forward (Supports Deep Supervision)
                output = model(images)
                
                # [🔥 CRITICAL UPDATE: Unpacking Logic]
                stu_feats = None
                final_pred = None # Used for metrics calculation
                preds_for_loss = None # Passed to loss function

                if isinstance(output, tuple):
                    if len(output) == 3:
                        # New: (Refined, Coarse, Feats)
                        refined, coarse, stu_feats = output
                        # Pack prediction tuple for Deep Supervision in Loss
                        preds_for_loss = (refined, coarse)
                        final_pred = refined # Use Refined for metrics
                    elif len(output) == 2:
                        # Old: (Preds, Feats) - Legacy support
                        preds_for_loss, stu_feats = output
                        final_pred = preds_for_loss
                else:
                    # Single output (should not happen in training with TSR-Net)
                    preds_for_loss = output
                    final_pred = output

                # 3. Calculate Loss (Deep Supervision handled inside loss_fn)
                loss, _, _, _ = loss_fn(preds_for_loss, masks, stu_feats, tea_feats)

            # --- Backward ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Metrics ---
            train_loss_epoch += loss.item()
            with torch.no_grad():
                # Sigmoid is applied here for metrics
                pred_final_prob = torch.sigmoid(final_pred)
                m_vals = calculate_matting_metrics(pred_final_prob, masks)
                for i in range(4): train_metrics[i] += m_vals[i]

            train_loop.set_postfix(loss=loss.item())

        scheduler.step()

        # Averages
        train_loss_epoch /= len(train_loader)
        train_metrics = [x / len(train_loader) for x in train_metrics]

        # ==========================
        #      Validation Phase
        # ==========================
        val_loss_epoch = 0
        val_metrics = [0, 0, 0, 0]
        model.eval()

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val  ]")

        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                # Validation Forward
                # Note: TSR-Net returns ONLY refined_logits in eval() mode
                output = model(images)
                
                # Handle cases where model might still return tuple in eval (defensive coding)
                if isinstance(output, tuple):
                    preds = output[0] 
                else:
                    preds = output
                
                with torch.amp.autocast('cuda'):
                    loss, _, _, _ = loss_fn(preds, masks) 
                
                val_loss_epoch += loss.item()
                
                pred_final_prob = torch.sigmoid(preds)
                m_vals = calculate_matting_metrics(pred_final_prob, masks)
                for i in range(4): val_metrics[i] += m_vals[i]

                val_loop.set_postfix(loss=loss.item())

        val_loss_epoch /= len(val_loader)
        val_metrics = [x / len(val_loader) for x in val_metrics]

        # --- Logging ---
        log_data = [epoch, optimizer.param_groups[0]['lr'], 
                    train_loss_epoch, *train_metrics, 
                    val_loss_epoch, *val_metrics]
        logger.log(log_data)

        # Update History
        history['train_loss'].append(train_loss_epoch); history['val_loss'].append(val_loss_epoch)
        history['train_mse'].append(train_metrics[0]); history['val_mse'].append(val_metrics[0])
        history['train_sad'].append(train_metrics[1]); history['val_sad'].append(val_metrics[1])
        history['train_grad'].append(train_metrics[2]); history['val_grad'].append(val_metrics[2])
        history['train_acc'].append(train_metrics[3]); history['val_acc'].append(val_metrics[3])

        print(f"📉 Epoch {epoch} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val IoU(Acc): {val_metrics[3]:.2f}%")

        # --- Save Best Model ---
        current_iou = val_metrics[3]
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! IoU: {best_iou:.2f}%")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

        if epoch % 5 == 0:
            plot_history(history['train_loss'], history['val_loss'], 
                         history['train_sad'], history['val_sad'],
                         history['train_grad'], history['val_grad'],
                         history['train_mse'], history['val_mse'],
                         history['train_acc'], history['val_acc'],
                         config.LOG_DIR)

if __name__ == "__main__":
    train()