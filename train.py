# train.py
# ─────────────────────────────────────────
# Training and validation logic
# ─────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from config import *


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def run_training(model, train_loader, val_loader, device):
    """Full training loop with scheduler and checkpointing."""
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0

    print("🚀 Starting Training...\n")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}"
          f"{'Val Loss':<14}{'Val Acc':<12}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader,
                                            optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader,
                                         criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            tag = '  ⭐ Best!'
        else:
            tag = ''

        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<14.2f}"
              f"{val_loss:<14.4f}{val_acc:<12.2f}{tag}")

    print(f"\n✅ Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    """Plots and saves training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ResNet18 — Satellite Image Classification',
                 fontsize=14, fontweight='bold')

    ax1.plot(train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(val_losses,   'r-o', label='Val Loss',   linewidth=2)
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs,   'r-o', label='Val Accuracy',   linewidth=2)
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Training curves saved!")
