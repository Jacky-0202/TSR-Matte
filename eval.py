# eval.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys
import argparse

# --- Import Project Modules ---
sys.path.append(os.getcwd())
from config import Config
from models.twinswinunet import TwinSwinUNet
from models.refiner import Refiner

# ==========================================
# 🔧 USER CONFIGURATION
# ==========================================
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/Matte"
CHECKPOINT_PATH = "./checkpoints/General_base_1024_Twin_Refined_20260203_2349/tsr_general_fp16.pth"

# Performance Settings
BATCH_SIZE = 8      # H200 can handle 16, 32, or even 64.
NUM_WORKERS = 8      # CPU threads for data loading.

TEST_SETS = [
    {
        'name': 'DIS-TE1',
        'img_dir': os.path.join(DATASET_ROOT, 'DIS5K/DIS-TE1/im'),
        'gt_dir':  os.path.join(DATASET_ROOT, 'DIS5K/DIS-TE1/gt')
    },
    {
        'name': 'HRS10K',
        'img_dir': os.path.join(DATASET_ROOT, 'HRS10K/TE-HRS10K/im'), 
        'gt_dir':  os.path.join(DATASET_ROOT, 'HRS10K/TE-HRS10K/gt') 
    }
]
# ==========================================

class MattingMetric:
    """Accumulates metrics (SAD, MSE)"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sad_sum_list = [] # Sum of Absolute Differences (Academic Standard)
        self.mae_list = []     # Mean Absolute Error (For Quick Check)
        self.mse_list = []     # Mean Squared Error

    def update(self, pred, gt):
        # Normalize to 0-1
        p = pred.astype(np.float32) / 255.0
        g = gt.astype(np.float32) / 255.0
        
        # Calculate metrics on the ORIGINAL resolution
        abs_diff = np.abs(p - g)
        
        self.sad_sum_list.append(np.sum(abs_diff))
        self.mae_list.append(np.mean(abs_diff))
        self.mse_list.append(np.mean((p - g) ** 2))

    def get_results(self):
        if not self.sad_sum_list: return None
        return {
            'SAD_SUM': np.mean(self.sad_sum_list),
            'MAE': np.mean(self.mae_list),
            'SAD_1k': np.mean(self.mae_list) * 1000, # Matches Training Log
            'MSE': np.mean(self.mse_list),
            'Count': len(self.sad_sum_list)
        }

class EvalDataset(Dataset):
    """
    Custom Dataset for Batch Evaluation.
    We do NOT load GT here to save RAM. We only load GT when calculating metrics.
    """
    def __init__(self, img_dir, gt_dir, img_size=1024):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Preprocessing (Resize to Model Input Size)
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Find GT Path (Handle extension mismatch)
        gt_name = os.path.splitext(img_name)[0] + '.png'
        gt_path = os.path.join(self.gt_dir, gt_name)
        if not os.path.exists(gt_path):
             gt_path = os.path.join(self.gt_dir, img_name)

        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            # Return dummy if corrupt
            return torch.zeros(3, self.img_size, self.img_size), "", ""

        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=image_rgb)
        img_tensor = aug['image']

        # Return: Tensor, GT Path (string), Image Name (string)
        return img_tensor, gt_path, img_name

def evaluate_dataset(model, refiner, dataset_cfg, device):
    print(f"\n📊 Evaluating: {dataset_cfg['name']} (Batch Size: {BATCH_SIZE})")
    
    # 1. Setup Dataset & DataLoader
    if not os.path.exists(dataset_cfg['img_dir']):
        print(f"Skipping {dataset_cfg['name']} (Path not found)")
        return None

    dataset = EvalDataset(dataset_cfg['img_dir'], dataset_cfg['gt_dir'], Config.IMG_SIZE)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    metric = MattingMetric()
    
    # 2. Batch Inference Loop
    model.eval()
    if refiner: refiner.eval()
    
    pbar = tqdm(dataloader, total=len(dataloader), bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    with torch.no_grad():
        for batch_imgs, batch_gt_paths, _ in pbar:
            # Move to GPU & FP16
            batch_imgs = batch_imgs.to(device).half()
            
            # Forward Pass
            coarse_prob = model(batch_imgs)
            if refiner:
                final_prob = refiner(batch_imgs, coarse_prob)
            else:
                final_prob = coarse_prob
            
            # Convert to Numpy (Batch, 1, H, W) -> (Batch, H, W)
            preds = final_prob.squeeze(1).float().cpu().numpy()
            
            # 3. Post-process & Calculate Metrics (Iterate inside batch)
            for i in range(len(preds)):
                pred_mask = preds[i]
                gt_path = batch_gt_paths[i]
                
                if not os.path.exists(gt_path): continue
                
                # Load GT to get Original Size
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is None: continue
                
                orig_h, orig_w = gt.shape[:2]
                
                # Resize Prediction back to Original Resolution
                pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                pred_255 = (pred_mask_resized * 255).astype(np.uint8)
                
                # Update Metrics
                metric.update(pred_255, gt)

            # Update Progress Bar
            current_res = metric.get_results()
            if current_res:
                pbar.set_postfix(mae=f"{current_res['MAE']:.4f}", log_sad=f"{current_res['SAD_1k']:.1f}")

    return metric.get_results()

def main():
    device = torch.device(Config.DEVICE)
    print(f"🚀 High-Speed Evaluation | Device: {device} | FP16: On")
    
    # 1. Load Models
    print("📦 Loading Models...")
    model = TwinSwinUNet().to(device)
    refiner = Refiner().to(device) if Config.USE_REFINER else None
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
    else: model.load_state_dict(checkpoint)
    model.half()
    
    if refiner and 'refiner_state_dict' in checkpoint:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
        refiner.half()

    # 2. Run Evaluation
    print("\n" + "="*85)
    print(f"{'Dataset':<10} | {'MAE':<8} | {'Log_SAD':<8} | {'Real_SAD (Sum)':<15} | {'MSE (1e-3)':<10}")
    print(f"{'(Name)':<10} | {'(0-1)':<8} | {'(MAE*1k)':<8} | {'(Academic)':<15} | {'(Stability)':<10}")
    print("-" * 85)
    
    for ds_cfg in TEST_SETS:
        res = evaluate_dataset(model, refiner, ds_cfg, device)
        if res:
            print(f"{ds_cfg['name']:<10} | {res['MAE']:<8.4f} | {res['SAD_1k']:<8.1f} | {res['SAD_SUM']:<15.1f} | {res['MSE']*1000:<10.2f}")

    print("="*85 + "\n")

if __name__ == '__main__':
    main()