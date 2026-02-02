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
        if len(data) == 0:
            return np.array([])
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
    Plots training curves with 'Best Epoch' markers.
    Saves the result as 'training_curves.png'.
    """
    # Convert inputs to numpy
    train_losses, val_losses = to_cpu_numpy(train_losses), to_cpu_numpy(val_losses)
    train_sad, val_sad = to_cpu_numpy(train_sad), to_cpu_numpy(val_sad)
    train_grad, val_grad = to_cpu_numpy(train_grad), to_cpu_numpy(val_grad)
    train_mse, val_mse = to_cpu_numpy(train_mse), to_cpu_numpy(val_mse)
    train_acc, val_acc = to_cpu_numpy(train_acc), to_cpu_numpy(val_acc)
    
    if len(train_losses) == 0:
        print("⚠️ No data to plot.")
        return

    epochs = range(1, len(train_losses) + 1)
    
    # Setup styling - makes plots look scientific and clean
    plt.style.use('bmh') 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    def plot_metric(ax, train_data, val_data, title, ylabel, color_val='orange', mark_best='min'):
        """
        Plots a metric and highlights the best validation epoch.
        mark_best: 'min' (for Loss, SAD...) or 'max' (for Accuracy)
        """
        # Plot lines
        if len(train_data) > 0:
            ax.plot(epochs, train_data, 'b-', alpha=0.6, linewidth=1.5, label='Train')
        if len(val_data) > 0:
            ax.plot(epochs, val_data, color_val, linestyle='-', linewidth=2, label='Val')
            
            # Find and mark the best epoch
            if mark_best == 'min':
                best_idx = np.argmin(val_data)
                best_val = val_data[best_idx]
                # Add red dot
                ax.scatter(best_idx + 1, best_val, c='red', s=50, zorder=5, marker='o')
                # Add text label
                ax.annotate(f"{best_val:.4f}", (best_idx + 1, best_val), 
                            xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, color='red', fontweight='bold')
            elif mark_best == 'max':
                best_idx = np.argmax(val_data)
                best_val = val_data[best_idx]
                ax.scatter(best_idx + 1, best_val, c='red', s=50, zorder=5, marker='o')
                ax.annotate(f"{best_val:.2f}%", (best_idx + 1, best_val), 
                            xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, color='red', fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    # --- 1. Total Loss ---
    plot_metric(axes[0, 0], train_losses, val_losses, 'Total Loss', 'Loss', 'red', mark_best='min')

    # --- 2. SAD (Key Metric) ---
    # Note in title that it is scaled to avoid confusion
    plot_metric(axes[0, 1], train_sad, val_sad, 'SAD (Sum of Abs Diff) [k-Scale]', 'SAD / 1000', 'magenta', mark_best='min')

    # --- 3. Gradient Error ---
    plot_metric(axes[0, 2], train_grad, val_grad, 'Gradient Error (Sharpness)', 'Grad (x1000)', 'green', mark_best='min')

    # --- 4. MSE ---
    plot_metric(axes[1, 0], train_mse, val_mse, 'MSE (Mean Squared Error)', 'MSE', 'orange', mark_best='min')

    # --- 5. Accuracy ---
    plot_metric(axes[1, 1], train_acc, val_acc, 'Pixel Accuracy (%)', 'Accuracy', 'cyan', mark_best='max')

    # --- 6. Remove Empty Slot ---
    fig.delaxes(axes[1, 2])

    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100) # dpi=100 for balance between file size and quality
    plt.close()
    print(f"📊 Training curves updated: {save_path}")