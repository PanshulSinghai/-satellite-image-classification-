# predict.py
# ─────────────────────────────────────────
# Single image prediction with confidence
# ─────────────────────────────────────────

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from config import CLASSES, IMAGE_SIZE, MEAN, STD, MODEL_SAVE_PATH


def load_model(model, device):
    """Loads saved model weights."""
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
    return model


def predict_image(model, image_path, device):
    """Predicts class of a single satellite image."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output    = model(tensor)
        probs     = torch.softmax(output, dim=1)[0]
        conf, idx = probs.max(0)
        predicted = CLASSES[idx.item()]

    # Display image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted}\nConfidence: {conf.item()*100:.2f}%',
              fontsize=12, fontweight='bold', color='green')
    plt.show()

    # Print all probabilities
    print(f"\n🤖 Predicted : {predicted}")
    print(f"📊 Confidence: {conf.item()*100:.2f}%\n")
    print("All class probabilities:")
    for cls, prob in zip(CLASSES, probs):
        bar = '█' * int(prob.item() * 30)
        print(f"  {cls:<25} {prob.item()*100:5.2f}%  {bar}")

    return predicted, conf.item()
