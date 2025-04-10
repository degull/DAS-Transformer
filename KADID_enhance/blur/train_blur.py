# train_blur.py
""" import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from blur_dataloader import get_blur_dataloader
from collaborate_blur_model import BlurBranchRestorationModel

import lpips
from pytorch_msssim import ssim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ✅ 학습 설정
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256

# ✅ 손실 함수
mse_loss = nn.MSELoss()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    lpips_loss = lpips.LPIPS(net='alex').to(device)
    lpips_loss.eval()

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

            # ✅ Loss 조합
            mse = mse_loss(restored, ref_img)
            ssim_val = 1 - ssim(restored, ref_img, data_range=1.0, size_average=True)
            lpips_val = lpips_loss(restored, ref_img).mean()
            loss = mse + 0.1 * ssim_val + 0.1 * lpips_val

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"✅ Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "collaborate_blur_model.pth")
    print("✅ 모델 저장 완료: collaborate_blur_model.pth")

if __name__ == "__main__":
    train()
 """

# (각 블러마다 encoder + decoder 전부 별도)
# train_blur.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from blur_dataloader import get_blur_dataloader
from models.collaborate_blur_model import BlurBranchRestorationModel

import lpips
from pytorch_msssim import ssim  # pip install pytorch-msssim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ✅ 학습 설정
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256

# ✅ 손실 함수들
mse_loss = nn.MSELoss()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlurBranchRestorationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ✅ LPIPS
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    lpips_loss.eval()

    dataloader = get_blur_dataloader(
        csv_path="E:/ARNIQA/ARNIQA/dataset/KADID10K/kadid10k.csv",
        img_dir="E:/ARNIQA/ARNIQA/dataset/KADID10K/images",
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

            mse = mse_loss(restored, ref_img)
            ssim_val = 1 - ssim(restored, ref_img, data_range=1.0, size_average=True)
            lpips_val = lpips_loss(restored, ref_img).mean()

            loss = mse + 0.1 * ssim_val + 0.1 * lpips_val
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"✅ Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "collaborate_blur_model.pth")
    print("✅ 모델 저장 완료: collaborate_blur_model.pth")

if __name__ == "__main__":
    train()
