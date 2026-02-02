# eval.py

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# --- Import Project Modules ---
from config import Config
from models.twinswinunet import TwinSwinUNet
from utils.dataset import MattingDataset
from utils.metrics import calculate_matting_metrics

def evaluate():
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    
    # 2. Setup Paths (Automatically uses TEST_SET from Config)
    test_root = os.path.join(Config.DATA_ROOT, Config.TEST_SET)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    
    print("\n" + "="*50)
    print(f"🔍 EVALUATION REPORT: {Config.TASK_NAME}")
    print(f"   Target Resolution: {Config.IMG_SIZE}")
    print(f"   Test Set Path:    {test_root}")
    print(f"   Checkpoint:       {checkpoint_path}")
    print("="*50)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # 3. Initialize Model
    model = TwinSwinUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"✅ Model loaded. Best SAD recorded: {checkpoint.get('best_sad', 'N/A'):.4f}")

    # 4. Initialize Test Dataset
    # Note: Target size matches training to ensure consistency
    dataset = MattingDataset(
        root_dir=test_root,
        mode='val', # Uses 'val' mode logic (Resize + Normalize)
        target_size=Config.IMG_SIZE,
        schema=Config.SCHEMA
    )
    
    # We use batch_size=1 for precise metric calculation
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )
    
    print(f"📊 Processing {len(dataset)} images...")

    # 5. Metrics Accumulators
    total_sad = 0
    total_mse = 0
    total_grad = 0
    total_acc = 0
    total_mae = 0 # Mean Absolute Error (L1)
    
    # 6. Evaluation Loop
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Inference (Returns Sigmoid output in eval mode)
            pred = model(images)
            
            # Calculate standard metrics
            mse, sad, grad, acc = calculate_matting_metrics(pred, masks)
            mae = F.l1_loss(pred, masks).item()
            
            total_sad += sad
            total_mse += mse
            total_grad += grad
            total_acc += acc
            total_mae += mae

    # 7. Aggregate Scores
    num_samples = len(dataset)
    avg_sad = total_sad / num_samples
    avg_mse = (total_mse / num_samples) * 1000 # Scaling MSE for readability
    avg_grad = total_grad / num_samples
    avg_acc = total_acc / num_samples
    avg_mae = total_mae / num_samples

    # 8. Print Results
    print("\n" + "✨ FINAL TEST RESULTS " + "✨")
    print("-" * 30)
    print(f"🏆 Average SAD:   {avg_sad:.4f}")
    print(f"🏆 Average MSE:   {avg_mse:.4f} (x10^-3)")
    print(f"🏆 Average MAE:   {avg_mae:.6f}")
    print(f"🏆 Average Grad:  {avg_grad:.4f}")
    print(f"🏆 Average Acc:   {avg_acc:.2f}%")
    print("-" * 30)
    
    # Save results to a text file
    result_file = os.path.join(Config.CHECKPOINT_DIR, f"test_results_{Config.TEST_SET}.txt")
    with open(result_file, "w") as f:
        f.write(f"Task: {Config.TASK_NAME}\n")
        f.write(f"Test Set: {Config.TEST_SET}\n")
        f.write(f"SAD: {avg_sad:.4f}\n")
        f.write(f"MSE: {avg_mse:.4f}\n")
        f.write(f"MAE: {avg_mae:.6f}\n")
        f.write(f"Grad: {avg_grad:.4f}\n")
        f.write(f"Accuracy: {avg_acc:.2f}%\n")
    
    print(f"📄 Full report saved to: {result_file}")

if __name__ == "__main__":
    evaluate()