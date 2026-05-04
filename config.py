# config.py
# ─────────────────────────────────────────
# Central configuration for the project
# ─────────────────────────────────────────

# Dataset
DATASET_PATH  = 'eurosat/EuroSAT'
CLASSES       = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
NUM_CLASSES   = len(CLASSES)

# Image settings
IMAGE_SIZE    = 224
MEAN          = [0.3444, 0.3803, 0.4078]
STD           = [0.2034, 0.1365, 0.1148]

# Training
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 0.001
DROPOUT       = 0.4
RANDOM_SEED   = 42

# Train / Val / Test split
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
# Test gets the remaining 0.15

# Paths
MODEL_SAVE_PATH = 'best_model.pth'
