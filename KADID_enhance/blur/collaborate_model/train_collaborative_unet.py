import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ✅ Collaborative UNet 구성
class StackFeaturePooling(nn.Module):
    def __init__(self, pooling_type='avg'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, feat_stack):
        if self.pooling_type == 'avg':
            pooled = torch.mean(feat_stack, dim=0, keepdim=True)
        elif self.pooling_type == 'max':
            pooled, _ = torch.max(feat_stack, dim=0, keepdim=True)
        else:
            raise NotImplementedError(f"Pooling type {self.pooling_type} not implemented")
        return pooled.expand_as(feat_stack)

class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

# CollaborativeUNet 수정
class CollaborativeUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, pooling_type='avg'):
        super().__init__()
        self.enc1 = UNetConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)  # ✅ 수정된 부분
        self.dec1 = UNetConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.pooling = StackFeaturePooling(pooling_type)
        self.fuse = nn.Conv2d(base_channels * 4, base_channels, 1)  # base_channels = 64 → 256→64


    def forward(self, x_stack):
        e1 = self.enc1(x_stack)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        pooled = self.pooling(d2)
        d2 = self.fuse(torch.cat([d2, pooled], dim=1))  # output: [N, 64, H, W]
        d1 = self.up1(d2)  # input: [N, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# ✅ Patch 생성
def extract_patches_from_image(image: torch.Tensor, patch_size=128, stride=128):
    _, H, W = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    nH, nW = patches.shape[1], patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
    return patches

# ✅ Dataset 정의
class KADIDBlurPatchDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, patch_size=128, stride=128):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.blur_codes = ['01', '02', '03']
        self.image_list = [row['dist_img'] for _, row in self.df.iterrows() if any(code in row['dist_img'] for code in self.blur_codes)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_list[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        patches = extract_patches_from_image(img, patch_size=self.patch_size, stride=self.stride)
        return patches  # [N, C, H, W]

# ✅ 학습 루프
def train_collaborative_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = KADIDBlurPatchDataset(
        csv_path="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv",
        img_dir="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images",
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = CollaborativeUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(10):  # epoch 수 조절 가능
        epoch_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            patch_stack = batch[0].to(device)
            patch_stack = patch_stack.squeeze(0)
            optimizer.zero_grad()
            output = model(patch_stack)
            loss = criterion(output, patch_stack)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"✅ Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/collab_unet_blur.pth")
    print("✅ 모델 저장 완료: weights/collab_unet_blur.pth")

if __name__ == "__main__":
    train_collaborative_unet()
