# 🛰️ Satellite Image Classification using ResNet18 + PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.07%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning project that classifies satellite images into 10 
land-use categories using **Transfer Learning** with a pretrained 
**ResNet18** model in PyTorch. Includes **Grad-CAM explainability** 
to visualize what the model focuses on when making predictions.

---

## 🎯 Results

| Metric              | Score    |
|---------------------|----------|
| Training Accuracy   | 93.22%   |
| Validation Accuracy | 94.89%   |
| **Test Accuracy**   | **94.07%** |

---

## 📊 Training Curves

<img width="2084" height="740" alt="training_curves" src="https://github.com/user-attachments/assets/d8ec3d3d-14f4-430c-b276-61c73c108807" />

---

## 🔍 Grad-CAM Explainability

Grad-CAM heatmaps show **exactly which regions** of the satellite 
image the model focused on to make its prediction.
- 🔴 **Red/Yellow** = high attention regions
- 🔵 **Blue** = ignored regions

<img width="1895" height="3643" alt="gradcam_visualization" src="https://github.com/user-attachments/assets/042d0940-b7ce-48bc-96c1-e0e5bc08f713" />


---

## 🗂️ Dataset

- **Name**: EuroSAT
- **Size**: 27,000 satellite images
- **Source**: [Kaggle — EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **Classes** (10 total):

| Class | Description |
|-------|-------------|
| AnnualCrop | Annual crop fields |
| Forest | Dense forest areas |
| HerbaceousVegetation | Herbaceous vegetation |
| Highway | Roads and highways |
| Industrial | Industrial buildings |
| Pasture | Pasture land |
| PermanentCrop | Permanent crop fields |
| Residential | Residential areas |
| River | Rivers and waterways |
| SeaLake | Sea and lake bodies |

---

## 🏗️ Model Architecture
  ResNet18 (Pretrained on ImageNet)
│
├── Conv layers 1-4  [FROZEN — keeps ImageNet features]
│
└── Custom Head [TRAINABLE]
├── Linear(512 → 256)
├── ReLU
├── Dropout(0.4)
└── Linear(256 → 10 classes)

### Why Transfer Learning?
Instead of training from scratch, we use ResNet18 pretrained on 
ImageNet and only train the final classification head. This gives us:
- ✅ Faster training
- ✅ Better accuracy with less data
- ✅ Avoids overfitting

---

## 📁 Project Structure

satellite-image-classification/
│
├── config.py          ← All settings and hyperparameters
├── dataset.py         ← Data loading, transforms, splitting
├── model.py           ← ResNet18 model definition
├── train.py           ← Training and validation functions
├── predict.py         ← Single image prediction
├── gradcam.py         ← Grad-CAM heatmap visualization
├── requirements.txt   ← Dependencies
│
├── training_curves.png       ← Loss & accuracy plots
├── gradcam_visualization.png ← Grad-CAM heatmaps
└── confusion_matrix.png      ← Class-wise performance

---

## 🚀 How to Run (Google Colab)

### Step 1 — Setup
```python
# Install dependencies
!pip install torch torchvision grad-cam matplotlib scikit-learn seaborn

# Download dataset from Kaggle
!kaggle datasets download -d apollo2506/eurosat-dataset
!unzip -q eurosat-dataset.zip -d eurosat
```

### Step 2 — Load Data
```python
from dataset import get_dataloaders
train_loader, val_loader, test_loader = get_dataloaders()
```

### Step 3 — Build Model
```python
import torch
from model import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = build_model(device)
```

### Step 4 — Train
```python
from train import run_training, plot_curves

train_losses, val_losses, train_accs, val_accs = run_training(
    model, train_loader, val_loader, device
)
plot_curves(train_losses, val_losses, train_accs, val_accs)
```

### Step 5 — Predict Single Image
```python
from predict import load_model, predict_image

model = load_model(model, device)
predicted, confidence = predict_image(model, 'path/to/image.jpg', device)
```

### Step 6 — Grad-CAM Visualization
```python
from gradcam import run_gradcam
run_gradcam(model, device, num_images=6)
```

---

## 💡 Key Concepts Used

| Concept | What it does |
|---------|-------------|
| Transfer Learning | Reuses ImageNet features, trains only final layer |
| Data Augmentation | Flips, rotations, color jitter to prevent overfitting |
| LR Scheduling | StepLR reduces learning rate every 5 epochs |
| Dropout (0.4) | Randomly drops neurons to prevent overfitting |
| Grad-CAM | Visualizes which image regions drive predictions |
| Model Checkpointing | Saves best model based on validation accuracy |

---

## 🔧 Tech Stack

- **PyTorch 2.x** — Deep learning framework
- **Torchvision** — ResNet18 + image transforms
- **pytorch-grad-cam** — Explainability heatmaps
- **Google Colab** — Free Tesla T4 GPU
- **EuroSAT** — Satellite image dataset

---

## 📈 Confusion Matrix
<img width="1118" height="989" alt="ConfusionMatrix" src="https://github.com/user-attachments/assets/e4abc30f-c9fe-49ad-8b76-4be8dce91c5e" />

---

## 👤 Author

Built as a portfolio project demonstrating:
- Transfer Learning with PyTorch
- Model Explainability with Grad-CAM  
- Real-world satellite image classification
