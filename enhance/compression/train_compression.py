# train_compression.py
import torch
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
