# utils/logger.py

import csv
import os
import torch

class CSVLogger:
    def __init__(self, save_dir, filename='training_log.csv'):
        """
        Args:
            save_dir (str): Directory where the log file will be saved.
            filename (str): Name of the CSV file.
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # [UPDATED] Expanded headers to include SAD and Grad
        # Matting metrics: MSE, SAD, Grad, Acc
        self.headers = [
            'Epoch', 'LR', 
            'Train_Loss', 'Train_MSE', 'Train_SAD', 'Train_Grad', 'Train_Acc', 
            'Val_Loss',   'Val_MSE',   'Val_SAD',   'Val_Grad',   'Val_Acc'
        ]
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file and write headers (overwrite mode 'w')
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        print(f"📝 Log file created at: {self.filepath}")

    def log(self, data):
        """
        Write a single row of data to the CSV.
        """
        clean_data = []
        
        for x in data:
            # 1. Handle PyTorch Tensors
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().item()
            
            # 2. Handle Floats
            if isinstance(x, float):
                # Use scientific notation for very small numbers
                if 0 < abs(x) < 0.0001:
                    clean_data.append(f"{x:.4e}")
                else:
                    # [UPDATED] increased precision slightly to .6f
                    clean_data.append(f"{x:.6f}")
            else:
                clean_data.append(x)

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(clean_data)