# utils/logger.py

import csv
import os
import torch

class CSVLogger:
    def __init__(self, save_dir, filename='training_log.csv', resume=False):
        """
        Initialize the CSV Logger.
        
        Args:
            save_dir (str): Directory where the log file will be saved.
            filename (str): Name of the CSV file.
            resume (bool): If True, append to existing file instead of overwriting.
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # Define CSV headers
        self.headers = [
            'Epoch', 'LR', 
            'Train_Loss', 'Train_MSE', 'Train_SAD', 'Train_Grad', 'Train_Acc', 
            'Val_Loss',   'Val_MSE',   'Val_SAD',   'Val_Grad',   'Val_Acc'
        ]
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # [Safety Logic] Handle Resume vs New Training
        file_exists = os.path.exists(self.filepath)
        
        if resume and file_exists:
            print(f"📝 Resuming logging to: {self.filepath}")
            # We don't write headers again if resuming
        else:
            # Overwrite mode: Create new file and write headers
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"📝 Created new log file at: {self.filepath}")

    def log(self, data):
        """
        Write a single row of data to the CSV.
        """
        clean_data = []
        
        for x in data:
            # 1. Handle PyTorch Tensors
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().item()
            
            # 2. Handle Floats (Unified formatting)
            if isinstance(x, (float, int)):
                if isinstance(x, float) and 0 < abs(x) < 1e-4:
                    clean_data.append(f"{x:.4e}") # Scientific notation for tiny numbers
                elif isinstance(x, float):
                    clean_data.append(f"{x:.6f}") # Standard precision
                else:
                    clean_data.append(x)
            else:
                clean_data.append(x)

        # Append data
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(clean_data)