# utils/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config

class MattingDataset(Dataset):
    """
    Universal Matting Dataset Loader.
    
    This class handles data loading for high-resolution matting tasks.
    It supports multiple dataset structures (Schemas) like DIS5K, COD10K, etc.,
    and applies rigorous data augmentation for training.
    """
    def __init__(self, root_dir, mode='train', target_size=1024, schema='standard'):
        """
        Args:
            root_dir (str): Path to the specific dataset split (e.g., '.../DIS5K/DIS-TR').
            mode (str): 'train' for augmentation, 'val' for deterministic loading.
            target_size (int): The target resolution for resizing (e.g., 1024).
            schema (str): Folder structure type ('standard', 'cod10k', 'hrsod').
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_size = target_size
        
        # [Refactoring Note]
        # We removed self.dilate_mask because we are strictly focusing on 
        # precise pixel-level matting, not morphological approximation.
        
        # =========================================================
        # 1. Define Folder Names based on Schema
        # =========================================================
        # Different datasets name their folders differently (e.g., 'gt', 'masks', 'GT').
        # This block unifies them into self.img_folder and self.mask_folder.
        if schema == 'standard':
            # Used by: DIS5K, Custom datasets
            # Structure: root/im, root/gt
            img_dir_name, gt_dir_name = 'im', 'gt'
        elif schema == 'cod10k':
            # Used by: COD10K
            # Structure: root/Image or root/Imgs, root/GT
            if os.path.exists(os.path.join(root_dir, 'Image')):
                img_dir_name, gt_dir_name = 'Image', 'GT'
            elif os.path.exists(os.path.join(root_dir, 'Imgs')):
                img_dir_name, gt_dir_name = 'Imgs', 'GT'
            else:
                # Fallback
                img_dir_name, gt_dir_name = 'im', 'gt'
        elif schema == 'hrsod':
            # Used by: HRSOD
            img_dir_name, gt_dir_name = 'imgs', 'masks'
        else:
            raise ValueError(f"Unknown schema: {schema}")

        # =========================================================
        # 2. Construct & Validate Paths
        # =========================================================
        self.img_folder = os.path.join(root_dir, img_dir_name)
        self.mask_folder = os.path.join(root_dir, gt_dir_name)
        
        # Sanity check to prevent obscure errors later during training
        if not os.path.exists(self.img_folder) or not os.path.exists(self.mask_folder):
            raise FileNotFoundError(
                f"❌ Dataset structure error! \n"
                f"   Schema: '{schema}' \n"
                f"   Looking for folders inside: {root_dir}\n"
                f"   - Images: {img_dir_name} (Found: {os.path.exists(self.img_folder)})\n"
                f"   - Masks:  {gt_dir_name} (Found: {os.path.exists(self.mask_folder)})\n"
                f"   Please check your config.py TRAIN_SET/VAL_SET paths."
            )
            
        # 3. Load File List
        # Filter only valid image extensions
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        self.image_files = sorted([f for f in os.listdir(self.img_folder) if f.lower().endswith(valid_ext)])
        
        print(f"[{mode.upper()}] Loaded {len(self.image_files)} images from: {self.img_folder}")

        # 4. Define Transformations (Albumentations)
        # We use Albumentations for fast and flexible image augmentation.
        if mode == 'train':
            self.transform = A.Compose([
                # --- Geometry Augmentations ---
                # Slight shifting, scaling, and rotating to make the model robust to position.
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                # Horizontal flipping is a standard augmentation for natural images.
                A.HorizontalFlip(p=0.5),
                
                # --- Color Augmentations ---
                # Adjust brightness/contrast to handle different lighting conditions.
                A.RandomBrightnessContrast(p=0.2),
                
                # --- Resizing & Padding ---
                # Resize the longest side to target_size (preserving aspect ratio)
                A.LongestMaxSize(max_size=target_size),
                # Pad the shorter side to center the image in a square canvas
                A.PadIfNeeded(
                    min_height=target_size, 
                    min_width=target_size, 
                    position='center',
                    border_mode=0,
                    value=0,      # Pad image with black
                    mask_value=0  # Pad mask with black
                ),
                
                # --- Normalization ---
                # Normalize using ImageNet mean/std (standard for Pretrained backbones)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Val/Test: Deterministic pipeline (No random flipping/scaling)
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size, 
                    min_width=target_size, 
                    position='center',
                    border_mode=0,
                    value=0,
                    mask_value=0
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. Read Image
        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        
        # Load image in RGB
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Warning: Could not read image {img_path}. Skipping.")
            return self.__getitem__((index + 1) % len(self)) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Read Mask
        # We support both .png and .jpg for masks. 
        # Note: Masks for matting/segmentation are usually single-channel grayscale.
        file_stem = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_folder, file_stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_folder, file_stem + '.jpg')
            
        mask = cv2.imread(mask_path, 0) # Read as Grayscale
        if mask is None:
            print(f"⚠️ Warning: Could not read mask for {img_name}. Skipping.")
            return self.__getitem__((index + 1) % len(self))

        # [Deleted Dilation Logic]
        # We no longer dilate the mask here. We want the model to learn the exact
        # ground truth boundaries, trusting the Twin Alignment + RRM strategy.

        # 3. Apply Augmentation
        # Albumentations handles both image and mask simultaneously
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        
        # Convert mask to Float tensor in range [0, 1]
        # Albumentations output is [H, W], we need [1, H, W] for PyTorch
        mask_tensor = augmented['mask'].float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return img_tensor, mask_tensor