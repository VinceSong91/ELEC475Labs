"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title("ELEC 475: Master Ablation Study (Validation Accuracy)", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-ablation", action="store_true")
    args = parser.parse_args()

    if args.plot_ablation:
        ablation_results = {
            "Base LeNet-5": {
                "val_accs": [47.10, 49.50, 49.79, 51.55, 52.78]
            },
            "Modern LeNet": {
                "val_accs": [52.20, 58.02, 60.69, 64.72, 66.66]
            },
            "AlexNet (No Norm)": {
                "val_accs": [
                    52.89, 59.12, 62.18, 64.54, 66.85,
                    68.21, 69.91, 71.05, 72.10, 72.89
                ]
            },
            "AlexNet (LRN)": {
                "val_accs": [
                    53.04, 59.48, 62.83, 65.20, 67.14,
                    68.73, 70.45, 71.51, 72.63, 73.24
                ]
            },
            "AlexNet (BatchNorm + Aug, 15 epochs)": {
                "val_accs": [
                    52.49, 59.57, 66.42, 68.38, 72.37,
                    69.71, 76.41, 75.51, 78.43, 79.62,
                    76.72, 78.24, 76.55, 83.50, 81.95
                ]
            },
            "AlexNet (BN, No Aug)": {
                "val_accs": [
                    46.87, 62.92, 62.79, 68.43, 72.91,
                    73.62, 73.73, 76.67, 77.05, 77.54,
                    77.67, 79.81, 79.44, 78.24, 81.66
                ]
            }
        }

        plot_ablation_summary(ablation_results)
