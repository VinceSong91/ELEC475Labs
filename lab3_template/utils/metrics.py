
"""
ELEC 475: Lab 3 - Localization Evaluation Metrics
"""

import os
import torch
from PIL import Image


def euclidean_errors(pred, gt):
    """
    Calculate Euclidean distance between predicted and true nose locations.

    Args:
        pred: (N, 2) predicted coordinates
        gt: (N, 2) ground truth coordinates

    Returns:
        (N,) tensor of Euclidean distances in pixels
    """
    return torch.sqrt(torch.sum((pred - gt) ** 2, dim=1))


def error_stats(errors):
    """
    Calculate summary statistics of localization errors.
    """
    return {
        "min": errors.min().item(),
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "max": errors.max().item(),
        "std": errors.std().item()
    }



def baseline_predictions(orig_size, train_file, img_dir):
    """
    Compute baseline predictions using the average normalized
    nose position from the training dataset.

    Returns:
        (N, 2) tensor of predictions in original-image pixels.
    """
    import os
    import re
    import torch
    from PIL import Image

    normalized_positions = []

    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Extract the filename and coordinate pair.
            match = re.match(
                r"""^\s*["']?(.+?\.(?:jpg|jpeg|png))["']?\s*[,:\s]+\s*["']?\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?["']?\s*$""",
                line,
                re.IGNORECASE
            )

            if match is None:
                raise ValueError(
                    f"Unrecognized training label format: {line!r}"
                )

            filename = match.group(1)
            u = float(match.group(2))
            v = float(match.group(3))

            image_path = os.path.join(img_dir, filename)

            with Image.open(image_path) as img:
                W, H = img.size

            normalized_positions.append([u / W, v / H])

    normalized_positions = torch.tensor(
        normalized_positions,
        dtype=torch.float32
    )

    mean_position = normalized_positions.mean(dim=0)

    # Convert normalized coordinates back into original pixels.
    return orig_size.float() * mean_position



def ced_curve(errors):
    """
    Calculate cumulative error distribution.

    Args:
        errors: (N,) tensor of localization errors

    Returns:
        x: sorted errors
        y: cumulative percentage of predictions
    """
    x, _ = torch.sort(errors)

    N = len(errors)

    y = torch.arange(
        1, N + 1, dtype=torch.float32, device=errors.device
    ) / N * 100

    return x, y


def best_worst_indices(errors):
    """
    Find indices of four best and four worst predictions.

    Args:
        errors: (N,) tensor of localization errors

    Returns:
        best: indices of four smallest errors
        worst: indices of four largest errors
    """
    sorted_indices = torch.argsort(errors)

    best = sorted_indices[:4]
    worst = sorted_indices[-4:].flip(0)

    return best, worst


def print_results(rows):
    """
    Print localization accuracy and computational cost tables.
    """

    print("\n" + "=" * 85)
    print("LOCALIZATION ERROR STATISTICS (pixels)")
    print("=" * 85)

    print(
        f"{'Model':<25}"
        f"{'Min':>12}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'Max':>12}"
        f"{'Std':>12}"
    )

    print("-" * 85)

    for name, row in rows.items():
        stats = error_stats(row["errors"])

        print(
            f"{name:<25}"
            f"{stats['min']:>12.1f}"
            f"{stats['mean']:>12.1f}"
            f"{stats['median']:>12.1f}"
            f"{stats['max']:>12.1f}"
            f"{stats['std']:>12.1f}"
        )

    print("\n" + "=" * 55)
    print("COMPUTATIONAL COST")
    print("=" * 55)

    print(
        f"{'Model':<25}"
        f"{'Params':>15}"
        f"{'ms/img':>15}"
    )

    print("-" * 55)

    for name, row in rows.items():
        print(
            f"{name:<25}"
            f"{row['params']:>15,}"
            f"{row['ms']:>15.1f}"
        )

    print("=" * 55)
