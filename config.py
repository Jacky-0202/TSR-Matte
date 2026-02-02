# config.py

import os
import torch
from datetime import datetime

class Config:
    # =========================================================================
    # 1. Core Task & Dataset Settings
    # =========================================================================
    TASK_NAME = 'DIS5K'  # Project Name for logging
    
    # [Dataset Paths]
    # Base root directory
    BASE_ROOT = "/home/tec/Desktop/Project/Datasets/Matte"
    
    # Auto-configure paths for DIS5K (Standard Structure)
    DATA_ROOT = os.path.join(BASE_ROOT, 'DIS5K')
    TRAIN_SET = 'DIS-TR'    # Path: DIS5K/DIS-TR/im
    VAL_SET   = 'DIS-VD'    # Path: DIS5K/DIS-VD/im
    TEST_SET  = 'DIS-TE1'   
    
    # Schema defines folder structure: 'standard' means im/gt
    SCHEMA    = 'standard'

    # =========================================================================
    # 2. Model Architecture Components
    # =========================================================================
    MODEL_TYPE = 'base'  # Options: 'tiny', 'small', 'base', 'large'

    # [Backbone Config]
    _BACKBONE_CONFIG = {
        'base': {
            'name': 'swin_base_patch4_window7_224', 
            'dim': 128, 
            'safe_size': 896 
        },
        'large': {
            'name': 'swin_large_patch4_window12_384_in22k', 
            'dim': 192, 
            'safe_size': 1152 
        },
    }
    
    if MODEL_TYPE not in _BACKBONE_CONFIG:
        raise ValueError(f"❌ Unknown MODEL_TYPE: {MODEL_TYPE}")

    _cfg = _BACKBONE_CONFIG[MODEL_TYPE]
    BACKBONE_NAME = _cfg['name']
    EMBED_DIM     = _cfg['dim']       # For Teacher Network
    SAFE_SIZE     = _cfg['safe_size'] # For Decoupled Strategy

    # [H200 Resolution Strategy]
    IMG_SIZE = 1024             # Target Output Resolution
    RESOLUTION_DECOUPLED = True # Resize to SAFE_SIZE for backbone, then restore

    # [Architecture Switches]
    # 1. Teacher Network (Twin Alignment): Guarantees structural integrity
    USE_TWIN_ALIGNMENT = True 
    
    # 2. Residual Refinement Module (RRM): Cleans up uncertainty halos
    USE_REFINER = True

    # =========================================================================
    # 3. Hardware Optimization (NVIDIA H200 Special)
    # =========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # [Memory & Speed]
    # H200 (141GB) allows massive batch sizes.
    # We use Batch=32 with Accum=1 for maximum throughput and stable BatchNorm.
    BATCH_SIZE = 2            
    GRAD_ACCUM_STEPS = 4       # Effective Batch Size = 32
    
    NUM_WORKERS = 16           # High CPU core count for fast data loading
    PIN_MEMORY = True         
    USE_AMP = True             # Automatic Mixed Precision (Essential for H200)

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    NUM_EPOCHS = 70          
    LR = 1e-4                 # Initial Learning Rate
    WEIGHT_DECAY = 1e-4       

    # =========================================================================
    # 5. Loss Function Weights (Matting Specialized)
    # =========================================================================
    # Note: Refiner (RRM) will automatically boost 'grad' weight internally.
    LOSS_WEIGHTS = {
        'bce': 0.0,       # Not used for Matting
        'dice': 0.0,      # Not used to avoid jagged edges
        'l1': 1.0,        # Pixel-level accuracy
        'grad': 1.0,      # Gradient Loss (Sharpness base)
        'ssim': 0.5,      # Structural Similarity (Texture)
        'feat': 0.2       # Feature Alignment (Student <-> Teacher)
    }

    # =========================================================================
    # 6. Output Paths & Logging
    # =========================================================================
    # Format: YYYYMMDD_HHMM (e.g., 20260202_1430)
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Tag generation for folder name
    _tag_twin = "Twin" if USE_TWIN_ALIGNMENT else "NoTwin"
    _tag_refine = "Refined" if USE_REFINER else "Base"
    
    CHECKPOINT_DIR = f"./checkpoints/{TASK_NAME}_{MODEL_TYPE}_{IMG_SIZE}_{_tag_twin}_{_tag_refine}_{_timestamp}"

    @staticmethod
    def print_info():
        print("\n" + "="*50)
        print(f"🚀 CONFIGURATION: {Config.TASK_NAME} (H200 Optimized)")
        print("-" * 50)
        print(f"   Mode:        Pure Matting (No Dilation)")
        print(f"   Pipeline:    Twin Alignment [{Config.USE_TWIN_ALIGNMENT}] + RRM [{Config.USE_REFINER}]")
        print(f"   Model:       {Config.MODEL_TYPE} ({Config.BACKBONE_NAME})")
        print(f"   Resolution:  {Config.IMG_SIZE} (Safe: {Config.SAFE_SIZE})")
        print("-" * 50)
        print(f"   Batch Size:  {Config.BATCH_SIZE} (Accum: {Config.GRAD_ACCUM_STEPS})")
        print(f"   Num Workers: {Config.NUM_WORKERS}")
        print(f"   Weights:     {Config.LOSS_WEIGHTS}")
        print(f"   Save Dir:    {Config.CHECKPOINT_DIR}")
        print("="*50 + "\n")