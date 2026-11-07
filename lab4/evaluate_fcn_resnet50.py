"""ELEC 475 Lab 4 pre-lab: evaluate pretrained FCN-ResNet50 on VOC2012.

Install: python -m pip install torch torchvision pillow numpy
Run:     python evaluate_fcn_resnet50.py --data-dir ./data
Quick:   python evaluate_fcn_resnet50.py --data-dir ./data --max-images 20
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

NUM_CLASSES = 21  # background + 20 VOC object categories
IGNORE_INDEX = 255  # unlabeled/void pixels in VOC segmentation masks
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)


def select_device(requested):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_collate(preprocess):
    # Keep each original mask at its own resolution. VOC images vary in size.
    def collate(batch):
        return [(preprocess(image), np.asarray(mask, dtype=np.int64).copy())
                for image, mask in batch]
    return collate


def update_confusion(confusion, predictions, targets):
    predictions = predictions.reshape(-1).astype(np.int64)
    targets = targets.reshape(-1).astype(np.int64)
    valid = ((targets != IGNORE_INDEX) & (targets >= 0) &
             (targets < NUM_CLASSES) & (predictions >= 0) &
             (predictions < NUM_CLASSES))
    counts = np.bincount(
        NUM_CLASSES * targets[valid] + predictions[valid],
        minlength=NUM_CLASSES * NUM_CLASSES,
    )
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data", help="VOC dataset root")
    parser.add_argument("--split", choices=["val", "train", "trainval"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional quick test; omit to evaluate full split")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")

    device = select_device(args.device)
    print(f"Device: {device}")

    # Official torchvision pretrained segmentation weights use 21 VOC labels.
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    # VOC2012 segmentation labels: 0=background, 1..20=classes, 255=ignore.
    dataset = VOCSegmentation(root=args.data_dir, year="2012",
                             image_set=args.split, download=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.workers, collate_fn=make_collate(preprocess))
    limit = min(len(dataset), args.max_images or len(dataset))
    print(f"Dataset: VOC2012/{args.split}; evaluating {limit} of {len(dataset)} images")
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    started = time.perf_counter()

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            image, target = batch[0]
            image = image.unsqueeze(0).to(device)
            logits = model(image)["out"]  # [1, 21, H_model, W_model]
            # Upsample scores to ORIGINAL ground-truth mask resolution.
            original_h, original_w = target.shape
            logits = F.interpolate(logits, size=(original_h, original_w),
                                   mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            update_confusion(confusion, prediction, target)
            if (index + 1) % 100 == 0 or index + 1 == limit:
                print(f"Processed {index + 1}/{limit} images")

    intersection = np.diag(confusion).astype(np.float64)
    union = (confusion.sum(axis=0) + confusion.sum(axis=1) - intersection)
    iou = np.divide(intersection, union, out=np.full(NUM_CLASSES, np.nan),
                    where=union > 0)
    miou = np.nanmean(iou) if np.any(union > 0) else float("nan")
    valid_pixels = confusion.sum()
    pixel_accuracy = intersection.sum() / valid_pixels if valid_pixels else float("nan")

    print("\nPer-class IoU (classes absent from both labels and predictions are N/A):")
    for name, value in zip(VOC_CLASSES, iou):
        print(f"  {name:14s}: {'N/A' if np.isnan(value) else f'{value * 100:.2f}%'}")
    print(f"\nMean IoU (mIoU): {miou * 100:.2f}%")
    print(f"Pixel accuracy:  {pixel_accuracy * 100:.2f}%")
    print(f"Valid pixels:    {valid_pixels:,} (void label 255 excluded)")
    print(f"Elapsed time:    {time.perf_counter() - started:.1f} seconds")
    if limit < len(dataset):
        print("Note: subset evaluation; run without --max-images for full-split mIoU.")


if __name__ == "__main__":
    main()
