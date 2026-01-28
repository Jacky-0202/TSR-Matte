# config.py

import torch
import os

# --- 1. Device & System ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4         # Number of data loading threads
PIN_MEMORY = True       # Accelerate data transfer to GPU

# --- 2. Model Selection ---
BACKBONE_OPTIONS = {
    'tiny':  'swin_tiny_patch4_window7_224',    # [96 dim]
    'small': 'swin_small_patch4_window7_224',   # [96 dim]
    'base':  'swin_base_patch4_window7_224',    # [128 dim]
    'base384': 'swin_base_patch4_window12_384', # [128 dim]
    'large': 'swin_large_patch4_window12_384'   # [192 dim]
}

MODEL_SELECT = 'base' 

if MODEL_SELECT not in BACKBONE_OPTIONS:
    raise ValueError(f"Invalid model selection: {MODEL_SELECT}")

BACKBONE_NAME = BACKBONE_OPTIONS[MODEL_SELECT]
print(f"🔹 Selected Backbone: {MODEL_SELECT.upper()} ({BACKBONE_NAME})")

# --- 3. Hyperparameters ---
IMG_SIZE = 1024         # 1024x1024 input
BATCH_SIZE = 2         # Adjust based on VRAM (8-16 for Base)
NUM_EPOCHS = 150        # Matting needs more epochs to refine details
LEARNING_RATE = 1e-4    
WEIGHT_DECAY = 1e-4     

# --- 4. Strategy Settings ---
# [Twin-Swin Strategy]
USE_TWIN_ALIGNMENT = True  
DILATE_MASK = False        # [CRITICAL CHANGE] Set to False for fine matting!

# --- 5. Loss Weights (Updated for Ultimate Matting Loss) ---
# These keys must match the __init__ arguments in utils/loss.py
LOSS_WEIGHTS = {
    'weight_struct': 1.0,  # [New] BiRefNet Structure Loss (Core)
    'weight_l1': 1.0,      # [New] L1 Loss for pixel precision
    'weight_grad': 1.0,    # [New] Gradient Loss for sharp edges
    'weight_ssim': 0.5,    # Structure consistency
    'weight_feat': 0.2     # Feature Alignment
}

# --- 6. Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATASET_ROOT = "Datasets/DIS5K_Flat" 
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/DIS5K_Flat" 

TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train') 
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')     

# --- 7. Logging & Saving ---
# Updated experiment name to reflect "Matte" instead of "Loc"
EXPERIMENT_NAME = f"TwinSwin_{MODEL_SELECT.capitalize()}_Matte1024"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', EXPERIMENT_NAME)
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', EXPERIMENT_NAME)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')