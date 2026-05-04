# model.py
# ─────────────────────────────────────────
# ResNet18 model with custom classification head
# ─────────────────────────────────────────

import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DROPOUT


def build_model(device):
    """
    Loads pretrained ResNet18 and replaces the final
    classification layer for NUM_CLASSES satellite classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all base layers — keep ImageNet features
    for param in model.parameters():
        param.requires_grad = False

    # Custom classification head for satellite imagery
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(256, NUM_CLASSES)
    )

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"✅ ResNet18 loaded")
    print(f"   Total params    : {total:,}")
    print(f"   Trainable params: {trainable:,}")

    return model
