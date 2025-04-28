# 왜곡 분류 세분화(class=25)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr

from dataset.dataset_kadid10k import KADID10KDataset
from model.slide_transformer import SlideTransformer  

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"
CHECKPOINT_DIR = "C:/Users/IIPL02/Desktop/NEW/checkpoints/class_25_kadid"
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_DAS-Transformer_KADID10K.pth")

# ✅ 폴더가 없다면 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_CLASSES = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # ✅ 데이터 로딩
    train_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ✅ 모델 초기화
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    # ✅ 손실 함수 및 최적화
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"✅ 학습 시작: {NUM_EPOCHS} Epochs, Batch Size: {BATCH_SIZE}, Classes: {NUM_CLASSES}")

    # ✅ 학습 루프
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
            outputs = model(dist_img, mode="train")
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

        srcc, _ = spearmanr(true_labels, pred_labels)
        plcc, _ = pearsonr(true_labels, pred_labels)

        print(f"📊 Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, "
              f"Accuracy: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

        # ✅ 모델 저장
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 모델 저장 완료: {checkpoint_path}")

    # ✅ 최종 모델 저장
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"🎯 학습 완료 & 최종 모델 저장 경로: {FINAL_MODEL_PATH}")
# 📊 Epoch 197 - Loss: 0.2121, Accuracy: 91.20%, SRCC: 0.9501, PLCC: 0.9499