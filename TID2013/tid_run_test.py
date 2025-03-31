# tid_run_test.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append("C:/Users/IIPL02/Desktop/NEW")  # ✅ 이거 반드시 포함

import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from dataset.dataset_tid2013 import TID2013Dataset
from model.slide_transformer import SlideTransformer

CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/TID2013/tid2013.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/TID2013/distorted_images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/tid2013_epoch_200.pth"

NUM_CLASSES = 6
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    test_dataset = TID2013Dataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_labels, pred_labels = [], []

    with torch.no_grad():
        for dist_img, labels in test_loader:
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)
            outputs = model(dist_img)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    print(f"📌 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    class_names = [
        "noise", "blur_denoise", "compression",
        "color_distortion", "brightness_contrast", "spatial_artifacts"
    ]
    print("📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of DAS-Transformer on TID2013 (6-Class)")
    plt.tight_layout()
    plt.show()
