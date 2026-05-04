# gradcam.py
# ─────────────────────────────────────────
# Grad-CAM heatmap visualization
# ─────────────────────────────────────────

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from config import CLASSES, DATASET_PATH, IMAGE_SIZE, MEAN, STD


def run_gradcam(model, device, num_images=6):
    """Generates Grad-CAM heatmaps for random satellite images."""

    # Enable gradients for last layer
    for param in model.layer4.parameters():
        param.requires_grad = True

    target_layer = [model.layer4[-1]]

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    fig, axes = plt.subplots(num_images, 3, figsize=(14, num_images * 4))
    fig.suptitle('Grad-CAM — What ResNet18 Sees in Satellite Images',
                 fontsize=16, fontweight='bold', y=1.01)

    for ax, title in zip(axes[0],
                         ['Original Image', 'Grad-CAM Heatmap', 'Overlay']):
        ax.set_title(title, fontsize=13, fontweight='bold', color='navy')

    for row in range(num_images):
        random_class = random.choice(CLASSES)
        class_path   = os.path.join(DATASET_PATH, random_class)
        image_path   = os.path.join(class_path,
                                    random.choice(os.listdir(class_path)))

        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np  = np.array(img_pil) / 255.0

        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output    = model(img_tensor)
            probs     = torch.softmax(output, dim=1)[0]
            conf, idx = probs.max(0)
            predicted = CLASSES[idx.item()]

        targets = [ClassifierOutputTarget(idx.item())]
        with GradCAM(model=model, target_layers=target_layer) as cam:
            grayscale = cam(input_tensor=img_tensor, targets=targets)[0]

        overlay = show_cam_on_image(img_np.astype(np.float32),
                                    grayscale, use_rgb=True)
        correct = (predicted == random_class)
        color   = 'green' if correct else 'red'
        symbol  = '✅' if correct else '❌'

        axes[row][0].imshow(img_pil)
        axes[row][0].axis('off')
        axes[row][0].set_ylabel(f'True: {random_class}', fontsize=10,
                                 rotation=0, labelpad=120, va='center')
        axes[row][1].imshow(grayscale, cmap='jet')
        axes[row][1].axis('off')
        axes[row][2].imshow(overlay)
        axes[row][2].axis('off')
        axes[row][2].set_title(
            f'{symbol} Pred: {predicted} ({conf.item()*100:.1f}%)',
            fontsize=10, color=color)

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Grad-CAM visualization saved!")
