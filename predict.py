# predict.py

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
# [UPDATED] Import the new TSRMatteNet
from models.tsr_matte import TSRMatteNet 

# --- 1. Settings ---
INPUT_DIR = 'test_data'         # Directory for input test images
OUTPUT_DIR = 'test_results'     # Directory for output results

# Batch Size for Inference
BATCH_SIZE = 8                  

# Model Path
MODEL_PATH = config.BEST_MODEL_PATH

# Inference Size (Height, Width)
IMG_SIZE = config.IMG_SIZE 

# --- 2. Inference Dataset ---
class InferenceDataset(Dataset):
    def __init__(self, root_dir, img_size, transform):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(valid_ext)])
        
        if len(self.files) == 0:
            print(f"⚠️ No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Resize to fixed size for batching
        img_resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply Transforms
        img_tensor = self.transform(img_resized)
        
        return img_tensor, img_name, orig_w, orig_h

# --- 3. Visualization Helper ---
def generate_checkerboard(h, w, tile_size=20):
    checkerboard = np.zeros((h, w, 3), dtype=np.uint8)
    color1 = (255, 255, 255) 
    color2 = (200, 200, 200) 
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                checkerboard[y:y+tile_size, x:x+tile_size] = color1
            else:
                checkerboard[y:y+tile_size, x:x+tile_size] = color2
    return checkerboard

def save_result(alpha_tensor, img_name, orig_w, orig_h, save_dir):
    """
    Saves the alpha matte restored to original size.
    """
    # alpha_tensor: (1, H, W) on GPU
    
    # 1. Upsample back to Original Size
    alpha_resized = F.interpolate(
        alpha_tensor.unsqueeze(0), 
        size=(orig_h, orig_w), 
        mode='bilinear', 
        align_corners=True
    )
    
    # 2. Convert to Numpy
    alpha_np = alpha_resized.squeeze().cpu().numpy() # (H, W)
    alpha_np = np.clip(alpha_np, 0, 1)
    
    # 3. Save Alpha
    save_base_name = os.path.splitext(img_name)[0]
    alpha_uint8 = (alpha_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, save_base_name + "_alpha.png"), alpha_uint8)

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' not found.")
        return

    device = config.DEVICE
    print(f"🚀 Loading Model from: {MODEL_PATH}")
    print(f"   Backbone: {config.BACKBONE}")
    print(f"   Inference Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")

    # [UPDATED] Initialize TSRMatteNet
    model = TSRMatteNet(
        n_classes=config.NUM_CLASSES, 
        img_size=config.IMG_SIZE, 
        backbone_name=config.BACKBONE, 
        pretrained=False
    ).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ Model weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print(f"❌ Weight file not found at: {MODEL_PATH}")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Prepare DataLoader ---
    test_dataset = InferenceDataset(INPUT_DIR, IMG_SIZE, transform)
    
    if len(test_dataset) == 0:
        return

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    print(f"📂 Found {len(test_dataset)} images. Processing in batches...")
    
    # --- Inference Loop ---
    with torch.no_grad():
        for batch_imgs, batch_names, batch_ws, batch_hs in tqdm(test_loader):
            batch_imgs = batch_imgs.to(device)
            
            # Forward Pass (Batch Inference)
            # gt_mask=None disables the Teacher branch
            preds, _, _ = model(batch_imgs, gt_mask=None)
            
            # Process each image in the batch
            for i in range(len(batch_imgs)):
                img_name = batch_names[i]
                orig_w = batch_ws[i].item()
                orig_h = batch_hs[i].item()
                alpha_pred = preds[i] # (1, H, W)
                
                # Restore size and Save
                save_result(alpha_pred, img_name, orig_w, orig_h, OUTPUT_DIR)

    print(f"\n✅ All Done! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()