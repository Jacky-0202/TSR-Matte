# predict.py

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse

# --- Import Project Modules ---
from config import Config
from models.twinswinunet import TwinSwinUNet
from models.refiner import Refiner  # [NEW] Import Refiner for 2nd stage

# ==========================================
# 🔧 USER CONFIGURATION
# ==========================================
# Path to your trained .pth file (Ensure this matches Config.IMG_SIZE)
CHECKPOINT_PATH = "./checkpoints/DIS5K_base_1024_Twin_Refined_20260202_1327/best_model.pth"

# Input: Can be a single image path OR a directory folder
INPUT_PATH = "/home/tec/Desktop/Project/Datasets/Matte/DIS5K/DIS-TE4/im" 

# Output: Where to save the results
OUTPUT_DIR = "./results"
# ==========================================

def parse_args():
    """Allows overriding paths via command line"""
    parser = argparse.ArgumentParser(description="TwinSwin-Matte Inference")
    parser.add_argument('--ckpt', type=str, default=CHECKPOINT_PATH, help='Path to checkpoint')
    parser.add_argument('--input', type=str, default=INPUT_PATH, help='Input image or folder')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Environment
    device = torch.device(Config.DEVICE)
    os.makedirs(args.output, exist_ok=True)
    
    Config.print_info()
    print(f"🚀 Inference Mode")
    print(f"   Model:      {Config.MODEL_TYPE}")
    print(f"   Refiner:    {'ON' if Config.USE_REFINER else 'OFF'}")
    print(f"   Checkpoint: {args.ckpt}")
    print(f"   Input:      {args.input}")
    print(f"   Output:     {args.output}\n")

    # 2. Load Models
    # A. Initialize Student (TwinSwin)
    model = TwinSwinUNet().to(device)
    
    # B. Initialize Refiner (RRM) if enabled in Config
    refiner_model = None
    if Config.USE_REFINER:
        print("✨ Initializing Refiner Network...")
        refiner_model = Refiner().to(device)

    # 3. Load Weights
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {args.ckpt}")

    print("📦 Loading weights...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # Load Student Weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load Refiner Weights (if applicable)
    if refiner_model is not None:
        if 'refiner_state_dict' in checkpoint:
            refiner_model.load_state_dict(checkpoint['refiner_state_dict'])
            print("✅ Refiner weights loaded successfully.")
            refiner_model.eval()
        else:
            print("⚠️ WARNING: Config.USE_REFINER is True, but checkpoint has no refiner weights!")
            print("   Running with initialized (random) Refiner weights... Result will be bad.")

    # 4. Define Preprocessing (Must match Training!)
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 5. Prepare File List
    if os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        image_paths.sort()
    else:
        image_paths = [args.input]

    if not image_paths:
        print(f"❌ No images found in {args.input}")
        return

    print(f"📂 Found {len(image_paths)} images. Processing...")

    # 6. Inference Loop
    for img_path in tqdm(image_paths):
        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Error reading: {img_path}")
            continue
        
        # Convert BGR (OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # Preprocess
        aug = transform(image=image)
        img_tensor = aug['image'].unsqueeze(0).to(device) # (1, 3, H, W)

        # Predict
        with torch.no_grad():
            # Step 1: Coarse Prediction (TwinSwin)
            # In eval mode, TwinSwin returns probability (0~1) directly
            coarse_prob = model(img_tensor) 
            
            # Step 2: Refinement (Refiner) - Optional
            if refiner_model is not None:
                # Refiner takes (RGB + Coarse Alpha) -> Refined Alpha
                final_prob = refiner_model(img_tensor, coarse_prob)
            else:
                final_prob = coarse_prob

            # Post-process
            # (1, 1, H, W) -> (H, W) -> CPU -> Numpy
            pred_mask = final_prob.squeeze().cpu().numpy()

        # Resize back to original image size
        # Using INTER_LINEAR for smoothness (don't use NEAREST for alpha!)
        pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Convert to 0-255 image
        pred_mask = (pred_mask * 255).astype(np.uint8)

        # Save Result
        filename = os.path.basename(img_path)
        save_name = os.path.splitext(filename)[0] + '.png'
        save_path = os.path.join(args.output, save_name)
        
        cv2.imwrite(save_path, pred_mask)

    print(f"\n✅ Done! Results saved to: {args.output}")

if __name__ == '__main__':
    main()