# utils/plot.py

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def to_cpu_numpy(data):
    """Helper to convert tensors/lists to numpy arrays safely."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return np.array([x.detach().cpu().numpy() for x in data])
        return np.array(data)
    return np.array(data)

def plot_history(train_losses, val_losses, train_sad, val_sad, train_grad, val_grad, train_mse, val_mse, train_acc, val_acc, save_dir):
    """
    Plots training curves: Loss, SAD, Grad, MSE, and Accuracy.
    """
    # Convert all inputs to numpy
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_sad = to_cpu_numpy(train_sad)
    val_sad = to_cpu_numpy(val_sad)
    train_grad = to_cpu_numpy(train_grad)
    val_grad = to_cpu_numpy(val_grad)
    train_mse = to_cpu_numpy(train_mse)
    val_mse = to_cpu_numpy(val_mse)
    train_acc = to_cpu_numpy(train_acc)
    val_acc = to_cpu_numpy(val_acc)
    
    epochs = range(1, len(train_losses) + 1)

    # Setup figure: 2 rows, 3 columns to fit 5 plots
    plt.figure(figsize=(18, 10))

    # --- 1. Total Loss ---
    plt.subplot(2, 3, 1)
    if len(train_losses) > 0:
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)

    # --- 2. SAD (The Most Important Metric) ---
    plt.subplot(2, 3, 2)
    if len(train_sad) > 0:
        plt.plot(epochs, train_sad, 'b-', label='Train SAD')
        plt.plot(epochs, val_sad, 'm-', label='Val SAD')
        plt.title('SAD (Sum of Absolute Differences) ↓')
        plt.xlabel('Epochs'); plt.ylabel('k-SAD')
        plt.legend(); plt.grid(True)

    # --- 3. Gradient Error (Edge Quality) ---
    plt.subplot(2, 3, 3)
    if len(train_grad) > 0:
        plt.plot(epochs, train_grad, 'b-', label='Train Grad')
        plt.plot(epochs, val_grad, 'g-', label='Val Grad')
        plt.title('Gradient Error (Edge Sharpness) ↓')
        plt.xlabel('Epochs'); plt.ylabel('Grad Error')
        plt.legend(); plt.grid(True)

    # --- 4. MSE ---
    plt.subplot(2, 3, 4)
    if len(train_mse) > 0:
        plt.plot(epochs, train_mse, 'b-', label='Train MSE')
        plt.plot(epochs, val_mse, 'orange', label='Val MSE')
        plt.title('MSE (Mean Squared Error) ↓')
        plt.xlabel('Epochs'); plt.ylabel('MSE')
        plt.legend(); plt.grid(True)

    # --- 5. Accuracy (Added) ---
    plt.subplot(2, 3, 5)
    if len(train_acc) > 0:
        plt.plot(epochs, train_acc, 'b-', label='Train Acc')
        plt.plot(epochs, val_acc, 'c-', label='Val Acc')
        plt.title('Pixel Accuracy % (Higher is Better) ↑')
        plt.xlabel('Epochs'); plt.ylabel('Acc (%)')
        plt.legend(); plt.grid(True)

    # Create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training curves saved at: {save_path}")