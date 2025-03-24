import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from dataset.dataset_live import LIVEDataset  # LIVE용 커스텀 Dataset
from model.slide_transformer import SlideTransformer

# ✅ 데이터 경로
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"  # 각 왜곡 폴더 포함된 상위 경로

# ✅ 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_CLASSES = 5  # ✅ LIVE는 5개 왜곡 유형
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # ✅ 데이터 로딩
    train_dataset = LIVEDataset(IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ✅ 모델 정의
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    # ✅ 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"✅ 학습 시작: {NUM_EPOCHS} Epochs, Batch Size: {BATCH_SIZE}")

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

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

        # ✅ 모델 저장
        checkpoint_path = f"C:/Users/IIPL02/Desktop/NEW/checkpoints/LIVE_DAS-Transformer_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 모델 저장 완료: {checkpoint_path}")

    # ✅ 최종 모델 저장
    final_model_path = "C:/Users/IIPL02/Desktop/NEW/class=5_DAS-Transformer_LIVE.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"🎯 DAS-Transformer 학습 완료 & 최종 모델 저장 완료! 경로: {final_model_path}")
