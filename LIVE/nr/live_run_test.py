import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_live import LIVEDistortionDataset
from model.slide_transformer import SlideTransformer

CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/LIVE/live_dmos_full.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/live/LIVE_DAS_epoch_192.pth"

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    test_dataset = LIVEDistortionDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

    class_names = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]
    print("📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - DAS-Transformer on LIVE")
    plt.tight_layout()
    plt.show()
