# config.py

import torch
import os

# --- 1. Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Model Selection ---
# List of available Swin Transformer backbones.
SWIN_VARIANTS = [
    'swin_tiny_patch4_window7_224',    # [0] Tiny
    'swin_small_patch4_window7_224',   # [1] Small
    'swin_base_patch4_window7_224',    # [2] Base (Recommended for TSR-Matte)
    'swin_base_patch4_window12_384',   # [3] Base 384 (Heavy VRAM)
    'swin_large_patch4_window12_384'   # [4] Large (Overkill)
]

# 👉 Set to 2 (Swin Base) for high performance
MODEL_IDX = 2

try:
    BACKBONE = SWIN_VARIANTS[MODEL_IDX]
except IndexError:
    print(f"⚠️ Invalid MODEL_IDX: {MODEL_IDX}, defaulting to [0] Tiny")
    BACKBONE = SWIN_VARIANTS[0]

# --- 3. Hyperparameters ---
IMG_SIZE = 1024        # SOTA Standard for DIS5K
BATCH_SIZE = 2         # Set to 2 for Swin Base + 1024 size (Increase if VRAM > 24GB)
NUM_CLASSES = 1        
NUM_EPOCHS = 150       
NUM_WORKERS = 4        
PIN_MEMORY = True      

LEARNING_RATE = 2e-4   
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2
SCHEDULER_ETA_MIN = 1e-6

# --- 4. Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset Root Directory (Adjust if needed)
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/DIS5K_Flat"

# --- 5. Checkpoints & Logging ---
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Extract tag (e.g., "Base", "Tiny")
model_tag = 'Unknown'
if 'tiny' in BACKBONE: model_tag = 'Tiny'
elif 'small' in BACKBONE: model_tag = 'Small'
elif 'base' in BACKBONE: model_tag = 'Base'
elif 'large' in BACKBONE: model_tag = 'Large'

if '384' in BACKBONE:
    model_tag += '_384'

# [UPDATED] Official Experiment Name: TSR-Matte
# Format: TSR-Matte_SwinBase_1024
EXPERIMENT_NAME = f"TSR-Matte_Swin{model_tag}_{IMG_SIZE}"

SAVE_DIR = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
LAST_MODEL_PATH = os.path.join(SAVE_DIR, 'last_model.pth')

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)