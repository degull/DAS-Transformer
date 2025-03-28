import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append("C:/Users/IIPL02/Desktop/NEW")  # ✅ 이거 반드시 포함

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset_csiq_ref import CSIQDatasetFR
from model.slide_transformer_ref import SlideTransformerRef  # 👈 Full-Reference용 모델 사용
from scipy.stats import spearmanr, pearsonr

ROOT_DIR = "C:/Users/IIPL02/Desktop/NEW/data/CSIQ"
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

if __name__ == "__main__":
    train_dataset = CSIQDatasetFR(ROOT_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SlideTransformerRef(img_size=224, num_classes=NUM_CLASSES).to(DEVICE)

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

        for ref_img, dist_img, labels in train_loader:
            ref_img, dist_img, labels = ref_img.to(DEVICE), dist_img.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ref_img, dist_img)
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

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}")

        save_path = f"C:/Users/IIPL02/Desktop/NEW/checkpoints/csiq/fr_6class_DAS-Transformer_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 저장됨: {save_path}")
