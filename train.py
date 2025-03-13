import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_kadid10k import KADID10KDataset

# dataset 폴더를 import 경로에 추가
sys.path.append(r"C:\Users\IIPL02\Desktop\NEW\dataset")
from model.slide_transformer import SlideTransformer  # 올바른 모듈명 적용

# 데이터 경로
CSV_PATH = r"C:\Users\IIPL02\Desktop\NEW\data\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\NEW\data\KADID10K\images"

# 하이퍼파라미터 설정
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':  # ✅ Windows multiprocessing 문제 해결!
    # 데이터셋 로드
    train_dataset = KADID10KDataset(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # DAS-Transformer 모델 로드
    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 학습 루프
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (dist_img, ref_img, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        
        # 에포크 별 정확도 출력
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "DAS-Transformer_KADID10K.pth")
    print("DAS-Transformer 학습 완료 & 저장 완료!")
