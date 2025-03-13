# 3/13

""" import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer  

CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    train_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        for batch_idx, (dist_img, labels) in enumerate(train_loader):  # ✅ _ 제거
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(dist_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        acc = 100. * correct / total

        # SRCC, PLCC 계산
        srcc, _ = spearmanr(true_labels, pred_labels)
        plcc, _ = pearsonr(true_labels, pred_labels)

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    torch.save(model.state_dict(), "3/13_DAS-Transformer_KADID10K.pth")
    print("DAS-Transformer 학습 완료 & 저장 완료!")
 """

# 3/14
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.stats import spearmanr, pearsonr
from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer  
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# 데이터 경로
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

# 하이퍼파라미터 설정
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ float 변환을 위한 Custom Transform 클래스
class ToFloatTransform:
    def __call__(self, image, **kwargs):
        return image.astype(np.float32) / 255.0  # 🔥 float32 변환 (0~1 정규화)

# ✅ Windows `multiprocessing` 문제 방지를 위해 추가
if __name__ == '__main__':
    # Albumentations 기반 데이터 변환 (🔥 Lambda 제거 → Custom Transform 적용)
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.ToFloat(max_value=255),  # ✅ float 변환 (권장)
        ToTensorV2()
    ])

    # 데이터셋 로드
    train_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=train_transform)

    # 🔥 클래스 불균형 해결 - Weighted Sampling 적용
    class_counts = [810, 1215, 810, 1215, 810, 5265]  
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    sample_weights = [weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # ✅ Windows에서는 multiprocessing 문제를 피하기 위해 `num_workers=0`으로 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    # 모델 로드
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    # 🔥 Focal Loss 적용
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss()(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 학습 루프
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        true_labels = []
        pred_labels = []

        for batch_idx, (dist_img, labels) in enumerate(train_loader):
            dist_img, labels = dist_img.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(dist_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        acc = 100. * correct / total

        # SRCC, PLCC 계산
        srcc, _ = spearmanr(true_labels, pred_labels)
        plcc, _ = pearsonr(true_labels, pred_labels)

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

    torch.save(model.state_dict(), "3/14_DAS-Transformer_KADID10K.pth")
    print("🔥 DAS-Transformer 학습 완료 & 저장 완료! 🚀")
