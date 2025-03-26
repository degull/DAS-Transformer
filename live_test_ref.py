import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os

from dataset.dataset_live_ref import LIVEDistortionDatasetRef
from model.slide_transformer_ref import SlideTransformerRef

# ✅ 설정 경로
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/LIVE/live_dmos_full.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/live_ref/epoch_200.pth"  # 🔁 마지막 학습 모델 경로 사용
BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # ✅ 데이터셋 및 DataLoader 준비
    test_dataset = LIVEDistortionDatasetRef(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ✅ 모델 로드
    model = SlideTransformerRef(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ✅ 예측 및 정답 저장 리스트
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for dist_img, ref_img, labels in test_loader:
            dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
            outputs = model(dist_img, ref_img)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # ✅ 평가 지표 출력
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    print(f"📊 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    # ✅ 클래스 이름 정의
    class_names = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]

    # ✅ Classification Report 출력
    print("📋 Classification Report:\n")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # ✅ Confusion Matrix 시각화
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - DAS-Transformer with Ref + Dist on LIVE")
    plt.tight_layout()
    plt.show()
