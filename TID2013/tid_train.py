# tid_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_tid2013 import TID2013Dataset
from model.slide_transformer import SlideTransformer
from scipy.stats import spearmanr, pearsonr

# 경로 설정
CSV_PATH = "C:/Users/IIPL02/Desktop/NEW/data/TID2013/tid2013.csv"
IMG_DIR = "C:/Users/IIPL02/Desktop/NEW/data/TID2013/distorted_images"

NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    train_dataset = TID2013Dataset(CSV_PATH, IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SlideTransformer(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"✅ TID2013 학습 시작")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        true_labels, pred_labels = [], []

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

        scheduler.step()
        acc = 100. * correct / total
        srcc, _ = spearmanr(true_labels, pred_labels)
        plcc, _ = pearsonr(true_labels, pred_labels)

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Acc: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")
        torch.save(model.state_dict(), f"C:/Users/IIPL02/Desktop/NEW/checkpoints/tid2013_epoch_{epoch+1}.pth")
