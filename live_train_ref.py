import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.stats import spearmanr, pearsonr
import os

from dataset.dataset_live_ref import LIVEDistortionDatasetRef
from model.slide_transformer_ref import SlideTransformerRef  # 👈 두 입력 받는 구조

# ✅ 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/LIVE/live_dmos_full.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/LIVE"
SAVE_DIR = "C:/Users/IIPL02/Desktop/NEW/checkpoints/live_ref"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ 학습 루프
if __name__ == '__main__':
    train_dataset = LIVEDistortionDatasetRef(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SlideTransformerRef(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"✅ LIVE 학습 시작: {NUM_EPOCHS} Epochs, Batch Size: {BATCH_SIZE}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        true_labels, pred_labels = [], []

        for dist_img, ref_img, labels in train_loader:
            dist_img, ref_img, labels = dist_img.to(DEVICE), ref_img.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(dist_img, ref_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

        scheduler.step()
        acc = 100. * correct / total
        srcc, _ = spearmanr(true_labels, pred_labels)
        plcc, _ = pearsonr(true_labels, pred_labels)

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Acc: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

        # ✅ 모델 저장
        model_save_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 모델 저장 완료: {model_save_path}")
