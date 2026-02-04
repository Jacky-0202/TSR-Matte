# export_inference.py

import torch
import os
import argparse
import sys

# Ensure the project modules can be imported
sys.path.append(os.getcwd())

from config import Config
from models.twinswinunet import TwinSwinUNet
from models.refiner import Refiner

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
# Path to your best training checkpoint (Contains Optimizer, Scheduler, etc.)
SOURCE_PATH = "./checkpoints/General_base_1024_Twin_Refined_20260203_2349/best_model.pth"

# Path for the output lightweight inference model
OUTPUT_PATH = "./checkpoints/General_base_1024_Twin_Refined_20260203_2349/inference_model_fp16.pth"
# ==========================================

def main():
    # 1. Validation
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError(f"❌ Source file not found: {SOURCE_PATH}")
    
    print(f"🔧 Processing: {SOURCE_PATH}")
    
    # 2. Load the Full Checkpoint
    # We load it to CPU first to avoid VRAM spikes during the conversion process.
    print("📦 Loading full checkpoint...")
    full_checkpoint = torch.load(SOURCE_PATH, map_location='cpu')

    # 3. Prepare a Clean Dictionary
    # This dictionary will only store what is strictly needed for inference.
    inference_checkpoint = {}

    # --- PROCESS STUDENT MODEL (TwinSwin) ---
    print("✂️  Stripping Optimizer & Converting Student to FP16...")
    
    # Initialize the model structure to match the weights
    student = TwinSwinUNet()
    
    # Handle potentially different state_dict formats (wrapped vs unwrapped)
    if 'state_dict' in full_checkpoint:
        student.load_state_dict(full_checkpoint['state_dict'])
    else:
        student.load_state_dict(full_checkpoint)
    
    # Convert weights to Half Precision (Float16)
    # This reduces memory usage and file size by ~50%.
    student.half()
    
    # Save the cleaned state_dict
    inference_checkpoint['state_dict'] = student.state_dict()

    # --- PROCESS REFINER MODEL (If enabled) ---
    if Config.USE_REFINER:
        if 'refiner_state_dict' in full_checkpoint:
            print("✂️  Stripping Refiner & Converting to FP16...")
            
            # Initialize Refiner
            refiner = Refiner()
            
            # Load weights
            refiner.load_state_dict(full_checkpoint['refiner_state_dict'])
            
            # Convert to FP16
            refiner.half()
            
            # Save the cleaned Refiner state_dict
            inference_checkpoint['refiner_state_dict'] = refiner.state_dict()
        else:
            print("⚠️  Warning: Config uses Refiner, but it was not found in the checkpoint.")

    # --- SAVE METADATA (Optional but recommended) ---
    # Storing config details helps identify the model architecture later.
    inference_checkpoint['meta'] = {
        'model_type': Config.MODEL_TYPE,
        'img_size': Config.IMG_SIZE,
        'sad_score': full_checkpoint.get('best_sad', 'N/A'), # Record the best score
        'description': 'Inference-only weights (FP16). Optimizer and History removed.'
    }

    # 4. Save to Disk
    print(f"💾 Saving to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(inference_checkpoint, OUTPUT_PATH)

    # 5. Print Statistics
    original_size = os.path.getsize(SOURCE_PATH) / (1024 * 1024)
    new_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    reduction = (1 - new_size / original_size) * 100
    
    print("\n" + "="*40)
    print("✅ Export Complete!")
    print(f"📂 Original (Train):   {original_size:.2f} MB")
    print(f"🚀 Optimized (Infer):  {new_size:.2f} MB")
    print(f"🔻 Size Reduction:     {reduction:.1f}%")
    print("="*40)

if __name__ == "__main__":
    main()