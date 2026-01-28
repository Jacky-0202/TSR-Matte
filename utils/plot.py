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
        # Handle empty lists
        if len(data) == 0:
            return np.array([])
        # Handle lists containing tensors
        if isinstance(data[0], torch.Tensor):
            return np.array([x.detach().cpu().item() for x in data])
        return np.array(data)
    return np.array(data)

def plot_history(train_losses, val_losses, 
                 train_sad, val_sad, 
                 train_grad, val_grad, 
                 train_mse, val_mse, 
                 train_acc, val_acc, 
                 save_dir):
    """
    Plots training curves: Loss, SAD, Grad, MSE, and Accuracy.
    Saves the result as 'training_curves.png'.
    """
    # Convert all inputs to numpy arrays
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
    
    # Prevent plotting errors if data is missing
    if len(train_losses) == 0:
        print("⚠️ No data to plot.")
        return

    epochs = range(1, len(train_losses) + 1)

    # Setup figure: 2 rows, 3 columns (Total 6 slots, last one will be removed)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Helper function for consistent plotting
    def plot_metric(ax, train_data, val_data, title, ylabel, color_val='orange'):
        if len(train_data) > 0:
            ax.plot(epochs, train_data, 'b-', label='Train')
        if len(val_data) > 0:
            ax.plot(epochs, val_data, color_val, linestyle='-', label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    # --- 1. Total Loss ---
    plot_metric(axes[0, 0], train_losses, val_losses, 'Total Loss', 'Loss', 'red')

    # --- 2. SAD (Key Metric) ---
    plot_metric(axes[0, 1], train_sad, val_sad, 'SAD (Sum of Abs Diff) ↓', 'k-SAD', 'magenta')

    # --- 3. Gradient Error ---
    plot_metric(axes[0, 2], train_grad, val_grad, 'Gradient Error (Edge Sharpness) ↓', 'Grad Error', 'green')

    # --- 4. MSE ---
    plot_metric(axes[1, 0], train_mse, val_mse, 'MSE (Mean Squared Error) ↓', 'MSE', 'orange')

    # --- 5. Accuracy ---
    plot_metric(axes[1, 1], train_acc, val_acc, 'Pixel Accuracy % (Higher is Better) ↑', 'Acc (%)', 'cyan')

    # --- 6. Remove the extra empty subplot (Row 2, Column 3) ---
    fig.delaxes(axes[1, 2])

    # Ensure directory exists and save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close figure to free memory
    print(f"📊 Training curves saved at: {save_path}")