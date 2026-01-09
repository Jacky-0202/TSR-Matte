# preprocess_dis5k.py

import os
import shutil
from tqdm import tqdm


SOURCE_ROOT = '/home/tec/Desktop/Project/Datasets/DIS5K'
TARGET_ROOT = '/home/tec/Desktop/Project/Datasets/DIS5K_Flat'

SPLIT_MAPPING = {
    'DIS-TR': 'train',
    'DIS-VD': 'val',
    'DIS-TE1': 'test',
    'DIS-TE2': 'test',
    'DIS-TE3': 'test',
    'DIS-TE4': 'test',
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def process_dis_subset(src_folder_name, target_split_name):
    print(f"\n🚀 Processing: {src_folder_name} -> {target_split_name}")
    
    base_src_path = os.path.join(SOURCE_ROOT, src_folder_name)
    src_im_dir = os.path.join(base_src_path, 'im')
    src_gt_dir = os.path.join(base_src_path, 'gt')
    
    tgt_im_dir = os.path.join(TARGET_ROOT, target_split_name, 'im')
    tgt_gt_dir = os.path.join(TARGET_ROOT, target_split_name, 'gt')
    
    create_dir(tgt_im_dir)
    create_dir(tgt_gt_dir)
    
    if not os.path.exists(src_im_dir) or not os.path.exists(src_gt_dir):
        print(f"⚠️  Skipping {src_folder_name}: 'im' or 'gt' folder missing.")
        return

    valid_ext = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(src_im_dir) if f.lower().endswith(valid_ext)]
    
    count = 0
    for img_name in tqdm(images, desc=f"Copying files"):
        file_stem = os.path.splitext(img_name)[0]
        
        src_img_path = os.path.join(src_im_dir, img_name)
        
        mask_name = file_stem + '.png'
        src_mask_path = os.path.join(src_gt_dir, mask_name)
        
        if not os.path.exists(src_mask_path):
            mask_name = file_stem + '.jpg'
            src_mask_path = os.path.join(src_gt_dir, mask_name)
            
        if os.path.exists(src_mask_path):
            shutil.copy2(src_img_path, os.path.join(tgt_im_dir, img_name))
            
            shutil.copy2(src_mask_path, os.path.join(tgt_gt_dir, mask_name))
            
            count += 1
        else:
            print(f"⚠️ Mask not found for: {img_name}")

    print(f"✅ Copied {count} pairs from {src_folder_name}")

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ Error: Source root not found at {SOURCE_ROOT}")
        return

    if os.path.exists(TARGET_ROOT):
        print(f"⚠️  Target directory exists: {TARGET_ROOT}")
        ans = input("Do you want to DELETE it and re-process? (y/n): ")
        if ans.lower() == 'y':
            shutil.rmtree(TARGET_ROOT)
            print("🗑️  Deleted old directory.")
        else:
            print("🔄 Merging into existing directory...")

    print(f"📂 Source: {SOURCE_ROOT}")
    print(f"📂 Target: {TARGET_ROOT}")
    
    for src, tgt in SPLIT_MAPPING.items():
        process_dis_subset(src, tgt)
        
    print("\n🎉 All Done! Structure is now clean.")
    print(f"Check your data at: {TARGET_ROOT}")

if __name__ == "__main__":
    main()