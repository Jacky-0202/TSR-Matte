import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- Import Project Modules ---
from config import Config
from models.twinswinunet import TwinSwinUNet

# ==========================================
# 🔧 USER CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "./checkpoints/TwinSwin_DIS5K_LOCATOR_1024_202601301620/best_model.pth"
INPUT_PATH = "/home/tec/Desktop/Project/Datasets/Matte/DIS5K/DIS-TE1/im" 
OUTPUT_DIR = "./results"
# ==========================================

def main():
    # 1. Setup Environment
    device = torch.device(Config.DEVICE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 Loading Model: {Config.MODEL_TYPE} ({Config.BACKBONE_NAME})")
    print(f"   Inference Resolution: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Input:      {INPUT_PATH}")
    print(f"   Output:     {OUTPUT_DIR}")

    # 2. Load Model
    model = TwinSwinUNet().to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {CHECKPOINT_PATH}")

    # Load Weights (Handle 'state_dict' wrapper if present)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval() # Set to evaluation mode (Freeze BN, Dropout)

    # 3. Define Preprocessing (Must match Training!)
    # We resize input to Target Size (1024), model handles Safe Size (896) internally.
    transform = A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Prepare File List
    if os.path.isdir(INPUT_PATH):
        image_paths = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [INPUT_PATH]

    if not image_paths:
        print(f"❌ No images found in {INPUT_PATH}")
        return

    print(f"📂 Found {len(image_paths)} images. Starting Inference...")

    # 5. Inference Loop
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
        img_tensor = aug['image'].unsqueeze(0).to(device) # (3, H, W) -> (1, 3, H, W)

        # Predict
        with torch.no_grad():
            # Model returns Sigmoid result (0~1) directly in eval mode
            pred_tensor = model(img_tensor) 
            
            # Post-process
            # (1, 1, H, W) -> (H, W) -> CPU -> Numpy
            pred_mask = pred_tensor.squeeze().cpu().numpy()

        # Resize back to original image size (Optional but recommended)
        # Using INTER_LINEAR for smoothness
        pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Convert to 0-255 image
        pred_mask = (pred_mask * 255).astype(np.uint8)

        # Save Result
        filename = os.path.basename(img_path)
        save_name = os.path.splitext(filename)[0] + '.png'
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        cv2.imwrite(save_path, pred_mask)

    print(f"✅ Done! Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()