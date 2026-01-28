# utils/logger.py

import csv
import os
import torch

class CSVLogger:
    def __init__(self, save_dir, filename='training_log.csv'):
        """
        Initialize the CSV Logger.
        
        Args:
            save_dir (str): Directory where the log file will be saved.
            filename (str): Name of the CSV file.
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # Define CSV headers
        # Metrics: Loss, MSE, SAD, Grad, Accuracy
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
        Data should match the order of self.headers.
        """
        clean_data = []
        
        for x in data:
            # 1. Handle PyTorch Tensors (detach to CPU)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().item()
            
            # 2. Handle Floats (Unified formatting)
            if isinstance(x, (float, int)):
                # Use scientific notation for very small numbers, else 6 decimal places
                if isinstance(x, float) and 0 < abs(x) < 1e-4:
                    clean_data.append(f"{x:.4e}")
                elif isinstance(x, float):
                    clean_data.append(f"{x:.6f}")
                else:
                    clean_data.append(x)
            else:
                clean_data.append(x)

        # Use 'a' (append) mode to ensure data is saved even if training crashes
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(clean_data)