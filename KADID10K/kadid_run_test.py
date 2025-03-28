# 3/13

""" import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer

# ✅ 데이터 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/1_3_13_DAS-Transformer_KADID10K.pth"

BATCH_SIZE = 32
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Windows에서 multiprocessing 오류 방지
if __name__ == '__main__':
    # ✅ 데이터셋 로드
    test_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 🔥 num_workers=0 설정

    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_labels = []
    pred_labels = []

    # ✅ 예측 실행
    with torch.no_grad():
        for dist_img, labels in test_loader:  # ✅ ref_img 제거하여 오류 해결
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)

            outputs = model(dist_img)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # ✅ 혼동 행렬 생성
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # ✅ SRCC, PLCC 계산
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)

    print(f"📌 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    # ✅ Classification Report 출력
    class_names = ["blur", "noise", "compression", "color", "contrast", "other"]
    print("📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    # ✅ 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of DAS-Transformer on KADID-10k")
    plt.show() """


# 3/14 (class = 6)
""" import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer

# ✅ 데이터 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/3_14_DAS-Transformer_KADID10K.pth"

BATCH_SIZE = 32
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Windows에서 multiprocessing 오류 방지
if __name__ == '__main__':
    # ✅ 데이터셋 로드 (6개 그룹으로 분류)
    test_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # 🔥 num_workers=0 설정

    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_labels = []
    pred_labels = []

    # ✅ 예측 실행
    with torch.no_grad():
        for dist_img, labels in test_loader:
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)

            outputs = model(dist_img)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # ✅ 혼동 행렬 생성
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # ✅ SRCC, PLCC 계산
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)

    print(f"📌 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    # ✅ Classification Report 출력
    class_names = ["blur", "noise", "compression", "color", "contrast", "other"]
    print("📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    # ✅ 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of DAS-Transformer on KADID-10k (6-Class)")
    plt.show()
 """

# 3/24 (class = 7)
import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
MODEL_PATH = "C:/Users/IIPL02/Desktop/NEW/checkpoints/7class_DAS-Transformer_epoch_199.pth"  # ✅ 7클래스 학습된 모델 경로

BATCH_SIZE = 32
NUM_CLASSES = 7  # ✅ 7개 그룹 기준
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ 테스트 시작
if __name__ == '__main__':
    # ✅ 데이터 로드
    test_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ✅ 모델 로드
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    true_labels = []
    pred_labels = []

    # ✅ 추론
    with torch.no_grad():
        for dist_img, labels in test_loader:
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)
            outputs = model(dist_img)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # ✅ SRCC & PLCC 계산
    srcc, _ = spearmanr(true_labels, pred_labels)
    plcc, _ = pearsonr(true_labels, pred_labels)
    print(f"📌 Test SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    # ✅ 클래스 이름 (7개 그룹)
    class_names = [
        "blur",
        "color_distortion",
        "compression",
        "noise",
        "brightness_change",
        "spatial_distortion",
        "sharpness_contrast"
    ]

    # ✅ Classification Report
    print("📌 Classification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))

    # ✅ Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # ✅ 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of DAS-Transformer on KADID-10k (7-Class)")
    plt.tight_layout()
    plt.show()
