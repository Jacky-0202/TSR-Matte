# utils/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class MattingDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=512):
        """
        Dataset for TwinSwin-Matte (DIS5K Structure).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        # 1. Construct Paths
        self.img_folder = os.path.join(root_dir, mode, 'im')
        self.mask_folder = os.path.join(root_dir, mode, 'gt')
        
        # 2. Get File List
        if not os.path.exists(self.img_folder):
            raise FileNotFoundError(f"Image folder not found: {self.img_folder}")
            
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.image_files = sorted([f for f in os.listdir(self.img_folder) if f.lower().endswith(valid_ext)])
        
        print(f"[{mode.upper()}] Loaded {len(self.image_files)} images from {self.img_folder}")

        # 3. Base Transforms (ImageNet Norm)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)
        
    def _round_to_32(self, x):
        """Helper to ensure dimensions are multiples of 32"""
        return int(round(x / 32) * 32)

    def __getitem__(self, index):
        # --- A. Load Image & Mask ---
        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        
        file_stem = os.path.splitext(img_name)[0]
        mask_name = file_stem + '.png'
        mask_path = os.path.join(self.mask_folder, mask_name)
        
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.mask_folder, file_stem + '.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"⚠️ Error loading {img_name}: {e}")
            # Return a dummy tensor or skip (handling skip inside __getitem__ is tricky, usually better to return a safe fallback)
            # Here we just re-raise to see the error, or you can implement a safe fallback.
            raise e

        # --- B. Augmentation & Transform ---
        if self.mode == 'train':
            # 1. Random Resize
            scale = np.random.uniform(0.5, 1.5)
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            
            # Resize
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.BILINEAR)

            # 2. Padding
            if new_w < self.img_size or new_h < self.img_size:
                pad_w = max(0, self.img_size - new_w)
                pad_h = max(0, self.img_size - new_h)
                image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
                mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=0)
            
            # 3. Random Rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-30, 30)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.BILINEAR)

            # 4. Random Crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
            # 5. Random Horizontal Flip
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 6. Gaussian Blur
            if np.random.random() > 0.5:
                sigma = np.random.uniform(0.1, 2.0)
                image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)

            # 7. Random Grayscale
            if np.random.random() > 0.8:
                image = TF.to_grayscale(image, num_output_channels=3)

            # 8. Color Jitter
            if np.random.random() > 0.2:
                color_tf = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                image = color_tf(image)

        else:
            # === Validation Logic ===
            w, h = image.size
            
            # Resize logic: Ensure strictly multiples of 32
            # 1. Resize based on short side match
            scale_ratio = self.img_size / min(w, h)
            new_w = w * scale_ratio
            new_h = h * scale_ratio
            
            # 2. Round dimensions to nearest multiple of 32
            final_w = self._round_to_32(new_w)
            final_h = self._round_to_32(new_h)
            
            # Ensure at least 32x32
            final_w = max(32, final_w)
            final_h = max(32, final_h)
            
            image = TF.resize(image, (final_h, final_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (final_h, final_w), interpolation=Image.BILINEAR)

        # --- C. ToTensor & Normalize ---
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = TF.to_tensor(mask)
        
        return image, mask