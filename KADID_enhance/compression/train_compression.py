# train_compression.py

# VGG 기반 Perceptual Loss + MSE Loss 추가 X
""" import torch
import torch.optim as optim
import torch.nn as nn
from ast_compression_model import ASTCompressionRestoration
from compression_dataloader import get_compression_dataloader
import time  # 실행 시간 측정을 위한 모듈

# ✅ 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 실행 환경: {device}")

# ✅ 2. 데이터셋 로드
csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

print("🔹 데이터 로드 중...")
dataloader = get_compression_dataloader(csv_path, img_dir, batch_size=4)
print(f"✅ 데이터셋 로드 완료! 총 {len(dataloader.dataset)}개의 JPEG/JPEG2000 이미지 포함.")

# ✅ 3. 모델 초기화
print("🔹 모델 초기화 중...")
model = ASTCompressionRestoration().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
print("✅ 모델 초기화 완료.")

# ✅ 4. 학습 루프
num_epochs = 50
print(f"🔹 {num_epochs} Epoch 동안 학습을 시작합니다...")

start_time = time.time()  # 전체 학습 시간 측정 시작

for epoch in range(num_epochs):
    epoch_start = time.time()  # 각 epoch 실행 시간 측정
    model.train()
    total_loss = 0

    print(f"\n🚀 Epoch [{epoch+1}/{num_epochs}] 시작...")

    for batch_idx, (img, ref) in enumerate(dataloader):
        batch_start = time.time()  # 배치별 실행 시간 측정

        img, ref = img.to(device), ref.to(device)

        optimizer.zero_grad()
        restored = model(img)
        loss = criterion(restored, ref)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_time = time.time() - batch_start  # 배치별 소요 시간 측정
        print(f"   🟢 Batch {batch_idx+1}/{len(dataloader)} 완료 (Loss: {loss.item():.6f}, Time: {batch_time:.3f}s)")

    epoch_time = time.time() - epoch_start  # Epoch 소요 시간 측정
    avg_loss = total_loss / len(dataloader)

    print(f"✅ Epoch [{epoch+1}/{num_epochs}] 완료 | 평균 Loss: {avg_loss:.6f} | Epoch Time: {epoch_time:.2f}s")

total_time = time.time() - start_time  # 전체 학습 시간 측정
print(f"\n✅ 모든 Epoch 학습 완료! 총 소요 시간: {total_time:.2f}s")

# ✅ 5. 모델 저장
model_save_path = "compression_restoration.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ 모델 저장 완료: {model_save_path}")
 """

# VGG 기반 Perceptual Loss + MSE Loss 추가 O
# train_compression.py
# train_compression.py
import torch
import torch.nn as nn
import torch.optim as optim
from ast_compression_model import ASTCompressionRestoration
from compression_dataloader import get_compression_dataloader
from torchvision.models import vgg16
import time

# ✅ VGG 기반 Perceptual Loss 정의
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16]  # conv3_3까지 사용
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

# ✅ 1. 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 실행 환경: {device}")

# ✅ 2. 데이터셋 로드
csv_path = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv"
img_dir = "C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images"

print("🔹 데이터 로드 중...")
dataloader = get_compression_dataloader(csv_path, img_dir, batch_size=4)
print(f"✅ 데이터셋 로드 완료! 총 {len(dataloader.dataset)}개의 JPEG/JPEG2000 이미지 포함.")

# ✅ 3. 모델 및 손실 함수 초기화
print("🔹 모델 초기화 중...")
model = ASTCompressionRestoration().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()
perc_loss = VGGPerceptualLoss().to(device)
print("✅ 모델 초기화 완료.")

# ✅ 4. 학습 루프
num_epochs = 1
print(f"🔹 {num_epochs} Epoch 동안 학습을 시작합니다...")

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0

    print(f"\n🚀 Epoch [{epoch+1}/{num_epochs}] 시작...")

    for batch_idx, (img, ref, distortion_class) in enumerate(dataloader):
        batch_start = time.time()

        img = img.to(device)
        ref = ref.to(device)
        distortion_class = distortion_class.to(device)  # 그대로 유지 (배치 전체)

        optimizer.zero_grad()

        # ✅ class_id를 배치 단위로 모델에 넘김
        restored = model(img, class_id=distortion_class)

        # ✅ 손실 계산
        loss_mse = mse_loss(restored, ref)
        loss_perc = perc_loss(restored, ref)
        loss = loss_mse + 0.1 * loss_perc

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_time = time.time() - batch_start

        print(f"   🟢 Batch {batch_idx+1}/{len(dataloader)} 완료 (Loss: {loss.item():.6f}, Time: {batch_time:.3f}s)")


    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] 완료 | 평균 Loss: {avg_loss:.6f} | Epoch Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\n✅ 모든 Epoch 학습 완료! 총 소요 시간: {total_time:.2f}s")

# ✅ 5. 모델 저장
model_save_path = "compression_restoration_with_perceptual.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ 모델 저장 완료: {model_save_path}")
