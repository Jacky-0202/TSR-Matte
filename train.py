import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- Import Custom Modules ---
import config
from models.tsr_matte import TSRMatteNet 

from utils.dataset import MattingDataset
from utils.loss import MattingLoss
from utils.metrics import calculate_matting_metrics
from utils.logger import CSVLogger
from utils.plot import plot_history

def get_dataloaders():
    print(f"📂 Dataset Root: {config.DATASET_ROOT}")
    
    train_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='train',
        img_size=config.IMG_SIZE
    )
    
    val_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='val',
        img_size=config.IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    # Validation Batch Size must be 1
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader

def build_model_and_optimizer(device):
    print(f"🏗️ Building Model: TSRMatteNet ({config.BACKBONE})...")
    
    # [UPDATED] Use TSRMatteNet
    model = TSRMatteNet(
        n_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        backbone_name=config.BACKBONE,
        pretrained=True
    ).to(device)

    # --- Optimizer Setup ---
    # LM Encoder (Student & Teacher) parameters
    img_enc_ids = list(map(id, model.img_encoder.parameters()))
    gt_enc_ids = list(map(id, model.gt_encoder.parameters()))
    
    # Decoder & Refiner parameters (Everything else)
    decoder_params = filter(lambda p: id(p) not in img_enc_ids and id(p) not in gt_enc_ids, model.parameters())

    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': config.LEARNING_RATE},           # Decoder/Refiner: Normal LR
        {'params': model.img_encoder.parameters(), 'lr': config.LEARNING_RATE * 0.1} # Backbone: 0.1x LR
    ], weight_decay=1e-3)

    # --- Loss Function ---
    # weight_grad=1.0 ensures we train for edge sharpness
    criterion = MattingLoss(
        weight_bce=1.0, weight_l1=1.0, weight_ssim=0.5, weight_iou=0.5, weight_grad=2.0, weight_feat=0.2
    ).to(device)
    
    # --- Scheduler ---
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.SCHEDULER_T0, 
        T_mult=config.SCHEDULER_T_MULT, 
        eta_min=config.SCHEDULER_ETA_MIN
    )
    
    return model, optimizer, criterion, scheduler

def train_one_epoch(loader, model, optimizer, criterion, scaler, device, epoch):
    model.train()
    
    # Accumulators
    avg_loss = 0
    avg_mse = 0
    avg_sad = 0
    avg_grad = 0
    avg_acc = 0
    
    # [PRINT] Print LR explicitly before the progress bar
    current_scale = model.res_scale.item()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n🔵 Epoch {epoch} | Current LR: {current_lr:.2e} | Current Scale:{current_scale:.3f}")
    
    # [TQDM] Progress bar
    loop = tqdm(loader, desc="   Train", leave=True)
    
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Forward pass using TSRMatteNet
            pred_alpha, img_feats, gt_feats = model(images, gt_mask=masks)
            loss, loss_l1, loss_detail, loss_feat = criterion(pred_alpha, masks, img_feats, gt_feats)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        mse, sad, grad, acc = calculate_matting_metrics(pred_alpha, masks)
        
        # Update Averages
        avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
        avg_mse = (avg_mse * batch_idx + mse) / (batch_idx + 1)
        avg_sad = (avg_sad * batch_idx + sad) / (batch_idx + 1)
        avg_grad = (avg_grad * batch_idx + grad) / (batch_idx + 1)
        avg_acc = (avg_acc * batch_idx + acc) / (batch_idx + 1)
        
        # Update TQDM - Only show Loss and Acc
        loop.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}%"
        )
        
    return avg_loss, avg_mse, avg_sad, avg_grad, avg_acc

def validate(loader, model, criterion, device, epoch):
    model.eval()
    
    avg_loss = 0
    avg_mse = 0
    avg_sad = 0
    avg_grad = 0
    avg_acc = 0
    
    loop = tqdm(loader, desc="   Valid", leave=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass without GT Mask (Teacher disabled)
            pred_alpha, img_feats, _ = model(images, gt_mask=None)
            loss, _, _, _ = criterion(pred_alpha, masks, img_feats, None)
            
            mse, sad, grad, acc = calculate_matting_metrics(pred_alpha, masks)
            
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            avg_mse = (avg_mse * batch_idx + mse) / (batch_idx + 1)
            avg_sad = (avg_sad * batch_idx + sad) / (batch_idx + 1)
            avg_grad = (avg_grad * batch_idx + grad) / (batch_idx + 1)
            avg_acc = (avg_acc * batch_idx + acc) / (batch_idx + 1)
            
            # Update TQDM - Only show Loss and Acc
            loop.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.2f}%"
            )
            
    return avg_loss, avg_mse, avg_sad, avg_grad, avg_acc

def main():
    if not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
        print(f"📁 Created Checkpoint Dir: {config.SAVE_DIR}")

    train_loader, val_loader = get_dataloaders()
    model, optimizer, criterion, scheduler = build_model_and_optimizer(config.DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    logger = CSVLogger(config.SAVE_DIR, filename='training_log.csv')
    
    best_sad = float('inf') 
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_sad': [], 'val_sad': [],
        'train_grad': [], 'val_grad': [],
        'train_mse': [], 'val_mse': [],
        'train_acc': [], 'val_acc': []
    }

    print(f"\n🚀 Start Training: {config.EXPERIMENT_NAME}")
    print(f"   Epochs: {config.NUM_EPOCHS} | Batch: {config.BATCH_SIZE} | Img Size: {config.IMG_SIZE}")
    print("-" * 60)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        
        # 1. Train
        t_loss, t_mse, t_sad, t_grad, t_acc = train_one_epoch(
            train_loader, model, optimizer, criterion, scaler, config.DEVICE, epoch
        )
        
        # 2. Validate
        v_loss, v_mse, v_sad, v_grad, v_acc = validate(
            val_loader, model, criterion, config.DEVICE, epoch
        )
        
        # 3. Scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log to CSV (still logs everything)
        logger.log([
            epoch, current_lr, 
            t_loss, t_mse, t_sad, t_grad, t_acc, 
            v_loss, v_mse, v_sad, v_grad, v_acc
        ])
        
        # 4. Save & Plot
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_sad'].append(t_sad);   history['val_sad'].append(v_sad)
        history['train_grad'].append(t_grad); history['val_grad'].append(v_grad)
        history['train_mse'].append(t_mse);   history['val_mse'].append(v_mse)
        history['train_acc'].append(t_acc);   history['val_acc'].append(v_acc)
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if v_sad < best_sad:
            best_sad = v_sad
            print(f"⭐ New Best Model (SAD: {best_sad:.4f}) Saved!")
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            
        plot_history(
            history['train_loss'], history['val_loss'],
            history['train_sad'], history['val_sad'],
            history['train_grad'], history['val_grad'],
            history['train_mse'], history['val_mse'],
            history['train_acc'], history['val_acc'],
            config.SAVE_DIR
        )

    print("\n✅ Training Completed!")

if __name__ == "__main__":
    main()