# dataset.py
# ─────────────────────────────────────────
# Data loading, transforms, and splitting
# ─────────────────────────────────────────

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import *


def get_transforms():
    """Returns train and test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, test_transform


def get_dataloaders():
    """Loads EuroSAT dataset and returns train/val/test dataloaders."""
    train_transform, test_transform = get_transforms()

    full_dataset = datasets.ImageFolder(root=DATASET_PATH)

    total      = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total)
    val_size   = int(VAL_SPLIT * total)
    test_size  = total - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_data.dataset.transform = train_transform
    val_data.dataset.transform   = test_transform
    test_data.dataset.transform  = test_transform

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    print(f"✅ Train: {train_size} | Val: {val_size} | Test: {test_size}")
    return train_loader, val_loader, test_loader
