
"""
ELEC 475: Lab 3 - Correction Model Dataset (Starter Template)
Builds training patches for the correction model from the training set only.

CorrectionDataset is a subclass of your Pre-Lab PetNoseDataset: it reuses its
parsed labels (self.samples) and image folder (self.img_dir), and replaces only
__getitem__.

Each sample is a 64 x 64 patch cut from a 128 x 128 region of the original image,
centred on a point near the true nose. The target is the offset from the patch
centre to the true nose, divided by HALF_PATCH so it lies roughly in [-1, 1]:

    corrected (u, v) = patch centre + target * HALF_PATCH
"""
import os
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from utils.dataset import PetNoseDataset, MEAN, STD

PATCH_SIZE = 128
HALF_PATCH = PATCH_SIZE // 2
INPUT_SIZE = 64
MAX_OFFSET = 40


def extract_patch(img: Image.Image, center) -> Image.Image:
    """
    Args:
        img    : PIL RGB image, original size
        center : (u, v) integer pixel coordinates of the patch centre.
                 The centre may lie near or outside the image border.

    Returns:
        PIL RGB image, INPUT_SIZE x INPUT_SIZE: the PATCH_SIZE x PATCH_SIZE region
        of img centred on center, resized with Image.BILINEAR. Any part of the
        region outside the image is black.
    """
    # ======================================================================
    # TODO [POST-LAB STEP 1]: Implement Patch Extraction
    #
    # Requirements:
    #   1. Crop the PATCH_SIZE x PATCH_SIZE region centred on center with img.crop()
    #      (img.crop() fills any part of the region outside the image with black)
    #   2. Resize the region to (INPUT_SIZE, INPUT_SIZE) with Image.BILINEAR
    # ======================================================================

    u, v = center

    left = u - HALF_PATCH
    top = v - HALF_PATCH
    right = u + HALF_PATCH
    bottom = v + HALF_PATCH

    patch = img.crop((left, top, right, bottom))

    patch = patch.resize(
        (INPUT_SIZE, INPUT_SIZE),
        resample=Image.BILINEAR
    )

    return patch


class CorrectionDataset(PetNoseDataset):
    """
    One random patch per training image per epoch. Inherits self.samples
    (a list of (name, u, v)), self.img_dir and __len__ from PetNoseDataset.
    """

    def __init__(
        self,
        root: str = "data/oxford-iiit-pet-noses",
        augment: bool = True
    ):
        super(CorrectionDataset, self).__init__(
            root=root,
            split="train"
        )

        self.augment = augment

        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        )

    def __getitem__(self, idx: int):
        """
        Returns (patch, target):
            patch  : torch.Tensor, shape (3, INPUT_SIZE, INPUT_SIZE), float,
                     TF.normalize(TF.to_tensor(patch), MEAN, STD)
            target : torch.Tensor, shape (2,), float. (du, dv) / HALF_PATCH, where (du, dv)
                     is the offset from the patch centre to the true nose, in pixels
        """
        # ======================================================================
        # TODO [POST-LAB STEP 1]: Implement Patch Sampling & Augmentation
        #
        # Requirements:
        #   1. Get (name, u, v) from self.samples[idx]; open self.img_dir / name as RGB
        #   2. Choose a random patch centre near the true nose: add a random offset,
        #      uniform in [-MAX_OFFSET, MAX_OFFSET], to u and to v, then round to integers
        #   3. Extract the patch centred on the patch centre with extract_patch(), and
        #      compute (du, dv) = true nose - patch centre
        #   4. If self.augment is True:
        #        a. With probability 0.5, flip the patch horizontally
        #           (patch.transpose(Image.FLIP_LEFT_RIGHT)) and update the target to match
        #        b. Apply self.color_jitter to the patch, and update the target if needed
        #   5. Return TF.normalize(TF.to_tensor(patch), MEAN, STD) and the
        #      normalized target (du, dv) / HALF_PATCH
        # ======================================================================

        # 1. Load image and true nose coordinates
        name, u, v = self.samples[idx]

        image_path = os.path.join(self.img_dir, name)

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # 2. Generate random patch centre near true nose
            offset_u = random.uniform(-MAX_OFFSET, MAX_OFFSET)
            offset_v = random.uniform(-MAX_OFFSET, MAX_OFFSET)

            center_u = round(u + offset_u)
            center_v = round(v + offset_v)

            center = (center_u, center_v)

            # 3. Extract patch
            patch = extract_patch(img, center)

        # Calculate offset from patch centre to true nose
        du = u - center_u
        dv = v - center_v

        # 4. Apply augmentations if enabled
        if self.augment:

            # Horizontal flip with probability 0.5
            if random.random() < 0.5:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

                # Horizontal flip reverses u offset
                du = -du

            # Colour jitter changes appearance, not coordinates
            patch = self.color_jitter(patch)

        # 5. Convert patch to normalized tensor
        patch_tensor = TF.to_tensor(patch)

        patch_tensor = TF.normalize(
            patch_tensor,
            MEAN,
            STD
        )

        # Normalize target offset by HALF_PATCH
        target = torch.tensor(
            [du / HALF_PATCH, dv / HALF_PATCH],
            dtype=torch.float32
        )

        return patch_tensor, target


if __name__ == "__main__":
    # Reality check: saves 8 patches with their centre (red cross) and
    # target nose position (green circle) drawn on.
    import matplotlib.pyplot as plt
    from utils.dataset import denormalize
    from utils.visualization import show_prediction

    centre = INPUT_SIZE / 2
    scale = INPUT_SIZE / PATCH_SIZE

    os.makedirs("plots", exist_ok=True)

    try:
        for augment in [False, True]:
            ds = CorrectionDataset(augment=augment)

            random.seed(0)

            fig, axes = plt.subplots(2, 4, figsize=(12, 6))

            for ax, i in zip(axes.flat, range(8)):
                patch, target = ds[i]

                offset = target * HALF_PATCH

                nose = (
                    centre + offset[0] * scale,
                    centre + offset[1] * scale
                )

                show_prediction(
                    ax,
                    TF.to_pil_image(denormalize(patch)),
                    nose,
                    (centre, centre),
                    f"offset ({offset[0]:.0f}, {offset[1]:.0f}) px"
                )

            save_path = os.path.join(
                "plots",
                f"correction_patches{'_aug' if augment else ''}.png"
            )

            fig.tight_layout()
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

            print(
                f"{len(ds):,} training images | "
                f"augment={augment} | "
                f"saved to: {save_path}"
            )

    except Exception as e:
        print(f"CorrectionDataset not fully implemented yet: {e}")
