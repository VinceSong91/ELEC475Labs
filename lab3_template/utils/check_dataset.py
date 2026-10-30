"""
Reality-check PetNoseDataset:
  1. Iterate every sample of both splits and verify image shape and label range.
  2. Plot eight training images with their nose labels and save to plots/dataset_check.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize, IMG_SIZE


def check_split(split: str) -> bool:
    ds = PetNoseDataset(split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    n_seen = 0
    bad_shape = 0
    bad_label = 0
    for images, labels, orig_sizes in loader:
        n_seen += images.shape[0]
        if images.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
            bad_shape += images.shape[0]
        inside = (labels >= 0) & (labels <= IMG_SIZE)   # both u and v in [0, 227]
        bad_label += (~inside.all(dim=1)).sum().item()

    ok = (n_seen == len(ds)) and bad_shape == 0 and bad_label == 0
    print(f"{split:5s}: {n_seen:,} samples | bad shapes: {bad_shape} "
          f"| labels outside frame: {bad_label} | {'OK' if ok else 'FAIL'}")
    return ok


def plot_samples(n: int = 8, path: str = "plots/dataset_check.png"):
    ds = PetNoseDataset(split="train")
    # evenly spaced indices for a varied sample
    idxs = torch.linspace(0, len(ds) - 1, n).long().tolist()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, i in zip(axes.flat, idxs):
        image, label, orig_size = ds[i]
        img = denormalize(image).permute(1, 2, 0).numpy()   # CHW -> HWC, in [0, 1]
        u, v = label.tolist()
        ax.imshow(img)
        ax.scatter([u], [v], c="red", s=40, edgecolors="white", linewidths=1)
        ax.set_title(f"{ds.samples[i][0]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    all_ok = all([check_split("train"), check_split("test")])
    plot_samples()
    print("All checks passed." if all_ok else "Some checks FAILED, see above.")