# utils/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class DIS5KDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=1024, dilate_mask=True):
        """
        DIS5K Dataset with Letterbox Resize (Keep Aspect Ratio).
        
        Args:
            root_dir (str): Path to 'Datasets/DIS5K_Flat/'
            mode (str): 'train' or 'val'
            target_size (int): The long-side resolution (e.g., 1024).
            dilate_mask (bool): Whether to apply slight dilation to GT (crucial for thin structures).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_size = target_size
        self.dilate_mask = dilate_mask
        
        # 1. Setup Paths
        # Structure: root/train/im/*.jpg, root/train/gt/*.png
        self.img_folder = os.path.join(root_dir, mode, 'im')
        self.mask_folder = os.path.join(root_dir, mode, 'gt')
        
        # 2. Check Paths
        if not os.path.exists(self.img_folder) or not os.path.exists(self.mask_folder):
            raise FileNotFoundError(f"Folder not found. Check: {self.img_folder}")
            
        # 3. Load File List
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.image_files = sorted([f for f in os.listdir(self.img_folder) if f.lower().endswith(valid_ext)])
        
        print(f"[{mode.upper()}] Found {len(self.image_files)} images in {self.img_folder}")

        # 4. Normalization (ImageNet stats)
        # Using standard mean/std for pre-trained backbones
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def _letterbox(self, img, mask, target_size):
        """
        Resize image and mask while keeping aspect ratio. 
        Pad the shorter side with zeros (black).
        """
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        # Image: Cubic/Linear for better quality
        # Mask: Linear/Area to preserve soft edges (if using soft labels) or Nearest
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # Or LINEAR if you want soft edges
        
        # Create canvas
        canvas_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Paste in center
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        
        canvas_img[start_y:start_y+new_h, start_x:start_x+new_w] = img_resized
        canvas_mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_resized
        
        return canvas_img, canvas_mask

    def __getitem__(self, index):
        # --- A. Load Data ---
        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        
        # Find corresponding mask
        file_stem = os.path.splitext(img_name)[0]
        mask_name = file_stem + '.png' # Masks are usually png
        mask_path = os.path.join(self.mask_folder, mask_name)
        
        # Fallback if mask has different extension
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.mask_folder, file_stem + '.jpg')

        # Read Image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Error reading image: {img_path}")
            return self.__getitem__((index + 1) % len(self)) # Skip broken
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Mask (Grayscale)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"⚠️ Error reading mask: {mask_path}")
            return self.__getitem__((index + 1) % len(self))

        # --- B. Pre-processing (Dilation) ---
        # Keep detail
        if self.dilate_mask and self.mode == 'train':
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # --- C. Augmentation (Train Only) ---
        if self.mode == 'train':
            # 1. Random Horizontal Flip
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # 2. Color Jitter (Simulated with simple numpy ops)
            if np.random.rand() > 0.2:
                # Brightness
                value = np.random.uniform(0.8, 1.2)
                image = np.clip(image * value, 0, 255).astype(np.uint8)

        # --- D. Letterbox Resize (The Core Logic) ---
        image, mask = self._letterbox(image, mask, self.target_size)

        # --- E. To Tensor & Normalize ---
        # Image: (H, W, 3) -> (3, H, W), Float32, 0-1, Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1) # HWC -> CHW
        image = torch.from_numpy(image).float()
        
        # Mask: (H, W) -> (1, H, W), Float32, 0-1
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0) # Add channel dim
        mask = torch.from_numpy(mask).float()
        
        return image, mask

# simple test
# if __name__ == "__main__":
#     root = "datasets_matte/DIS5K_Flat/"
#     if os.path.exists(root):
#         ds = DIS5KDataset(root, mode='train', target_size=1024)
#         img, mask = ds[0]
#         print(f"Image Shape: {img.shape}") # Should be (3, 1024, 1024)
#         print(f"Mask Shape: {mask.shape}") # Should be (1, 1024, 1024)