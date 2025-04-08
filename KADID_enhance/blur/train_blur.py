import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from blur_dataloader import get_blur_dataloader
from collaborate_blur_model import BlurBranchRestorationModel

# ✅ 학습 설정
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256

# ✅ 손실 함수
criterion = nn.MSELoss()

# ✅ 학습 루프
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = get_blur_dataloader(
        "E:/ARNIQA/ARNIQA/dataset/KADID10K/kadid10k.csv",
        "E:/ARNIQA/ARNIQA/dataset/KADID10K/images",
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (dist_img, ref_img, cls) in enumerate(dataloader):
            dist_img, ref_img, cls = dist_img.to(device), ref_img.to(device), cls.to(device)
            optimizer.zero_grad()
            restored = model(dist_img, class_id=cls)
            loss = criterion(restored, ref_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"✅ Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "collaborate_blur_model.pth")
    print("✅ 모델 저장 완료: collaborate_blur_model.pth")

if __name__ == "__main__":
    train()
