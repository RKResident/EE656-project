"""
datasets.py
-----------
Defines the PairedFERDataset, FERAugmentation transform, and helper
functions for building train / val DataLoaders.
"""

import random

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class FERAugmentation:
    """
    Random rotation (±15°), optional horizontal flip, resize to 112×112,
    convert to tensor, and normalise to [-1, 1].
    """

    def __init__(self, img_size: int = 112):
        self.angles = [-15, -10, -5, 0, 5, 10, 15]
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, img):
        angle = random.choice(self.angles)
        img = TF.rotate(
            img,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=(128, 128, 128),
        )
        if random.random() < 0.5:
            img = TF.hflip(img)

        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img


def get_eval_transform(img_size: int = 112) -> transforms.Compose:
    """Deterministic transform used for validation / inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairedFERDataset(Dataset):
    """
    Returns two augmented views of images that share the same emotion label.

    Parameters
    ----------
    samples : list of (path, label) tuples  – same format as
              ImageFolder.samples
    transform : callable applied to each PIL image
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

        # Build label → [indices] map
        self.label_to_indices: dict[int, list[int]] = {}
        for idx, (_, label) in enumerate(self.samples):
            self.label_to_indices.setdefault(label, []).append(idx)

        # Only keep samples whose class has ≥ 2 members (need a partner)
        self.valid_indices = [
            idx
            for idx, (_, label) in enumerate(self.samples)
            if len(self.label_to_indices[label]) >= 2
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img1_path, label = self.samples[real_idx]

        # Sample a *different* image with the same label
        candidates = self.label_to_indices[label]
        partner_idx = real_idx
        while partner_idx == real_idx:
            partner_idx = random.choice(candidates)
        img2_path, _ = self.samples[partner_idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 112,
) -> tuple[DataLoader, DataLoader, ImageFolder]:
    """
    Build paired train / val DataLoaders from ImageFolder-style directories.

    Returns
    -------
    train_loader, val_loader, train_imagefolder
        The ImageFolder object is returned so callers can access `.classes`.
    """
    train_imagefolder = ImageFolder(train_path)
    test_imagefolder = ImageFolder(test_path)

    train_transform = FERAugmentation(img_size=img_size)
    val_transform = get_eval_transform(img_size=img_size)

    train_dataset = PairedFERDataset(
        train_imagefolder.samples,
        transform=train_transform,
    )
    val_dataset = PairedFERDataset(
        test_imagefolder.samples,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_imagefolder