#  방식1 => deblurring 안됨
""" import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ✅ Stack Feature Pooling
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

# ✅ Basic UNet block
class UNetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

# ✅ Collaborative UNet (base_channels 증가)
class CollaborativeUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, pooling_type='avg'):
        super().__init__()
        self.enc1 = UNetConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = UNetConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.pooling = StackFeaturePooling(pooling_type)
        self.fuse = nn.Conv2d(base_channels * 4, base_channels, 1)

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
        d2 = self.fuse(torch.cat([d2, pooled], dim=1))
        d1 = self.up1(d2)
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

# ✅ Dataset
class KADIDBlurRefPatchDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, patch_size=128, stride=128):
        self.df = pd.read_csv(csv_path)
        self.dist_img_dir = img_dir
        self.ref_img_dir = img_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.blur_codes = ['01', '02', '03']
        self.image_pairs = [
            (row['dist_img'], row['ref_img']) for _, row in self.df.iterrows()
            if any(code in row['dist_img'] for code in self.blur_codes)
        ]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        dist_name, ref_name = self.image_pairs[idx]
        dist_path = os.path.join(self.dist_img_dir, dist_name)
        ref_path = os.path.join(self.ref_img_dir, ref_name)

        dist_img = Image.open(dist_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        dist_patches = extract_patches_from_image(dist_img, self.patch_size, self.stride)
        ref_patches = extract_patches_from_image(ref_img, self.patch_size, self.stride)
        return dist_patches, ref_patches

# ✅ 시각화 함수
def visualize_restoration(blur, restored, gt, index, save_dir="compare_results"):
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_pil(blur.cpu()))
    axes[0].set_title("Blurred")
    axes[0].axis("off")
    axes[1].imshow(to_pil(restored.cpu().clamp(0, 1)))
    axes[1].set_title("Restored")
    axes[1].axis("off")
    axes[2].imshow(to_pil(gt.cpu()))
    axes[2].set_title("Reference (GT)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_{index}.png"))
    plt.close()

# ✅ 학습 루프
def train_collaborative_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    dataset = KADIDBlurRefPatchDataset(
        csv_path="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv",
        img_dir="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images",
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = CollaborativeUNet(base_channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(200):
        total_loss = 0
        for i, (blur_stack, ref_stack) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            blur_stack = blur_stack.squeeze(0).to(device)
            ref_stack = ref_stack.squeeze(0).to(device)

            optimizer.zero_grad()
            restored = model(blur_stack)
            loss = criterion(restored, ref_stack)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch == 0 and i < 3:
                visualize_restoration(blur_stack[0], restored[0], ref_stack[0], index=f"{epoch}_{i}")

        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "C:/Users/IIPL02/Desktop/NEW/KADID_enhance/blur/collaborate_model/weights/collab_unet_blur_ref.pth")
    print("✅ 모델 저장 완료: weights/collab_unet_blur_ref.pth")

# ✅ 실행
if __name__ == "__main__":
    train_collaborative_unet()
 """

## 4/7
# 학습용 코드 (train_collaborative_unet.py)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class StackFeaturePooling(nn.Module):
    def __init__(self, pooling_type='avg'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, feat_stack):
        if self.pooling_type == 'avg':
            pooled = torch.mean(feat_stack, dim=0, keepdim=True)
        elif self.pooling_type == 'max':
            pooled, _ = torch.max(feat_stack, dim=0, keepdim=True)
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

class CollaborativeUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, pooling_type='avg'):
        super().__init__()
        self.enc1 = UNetConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = UNetConvBlock(base_channels * 2, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 1),
            nn.Sigmoid()  # [0,1] 범위로 제한
        )
        self.pooling = StackFeaturePooling(pooling_type)
        self.fuse = nn.Conv2d(base_channels * 4, base_channels, 1)

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
        d2 = self.fuse(torch.cat([d2, pooled], dim=1))
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

def extract_patches_from_image(image: torch.Tensor, patch_size=128, stride=128):
    _, H, W = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
    return patches

class KADIDBlurRefPatchDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, patch_size=128, stride=128):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.blur_codes = ['01', '02', '03']
        self.image_pairs = [
            (row['dist_img'], row['ref_img']) for _, row in self.df.iterrows()
            if any(code in row['dist_img'] for code in self.blur_codes)
        ]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        dist_name, ref_name = self.image_pairs[idx]
        dist_path = os.path.join(self.img_dir, dist_name)
        ref_path = os.path.join(self.img_dir, ref_name)

        dist_img = Image.open(dist_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        dist_patches = extract_patches_from_image(dist_img, self.patch_size, self.stride)
        ref_patches = extract_patches_from_image(ref_img, self.patch_size, self.stride)
        return dist_patches, ref_patches

def visualize_restoration(blur, restored, gt, index, save_dir="compare_results"):
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_pil(blur.cpu()))
    axes[0].set_title("Blurred")
    axes[1].imshow(to_pil(restored.cpu()))
    axes[1].set_title("Restored")
    axes[2].imshow(to_pil(gt.cpu()))
    axes[2].set_title("Reference (GT)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_{index}.png"))
    plt.close()

def train_collaborative_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    dataset = KADIDBlurRefPatchDataset(
        csv_path="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv",
        img_dir="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images",
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = CollaborativeUNet(base_channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(150):
        total_loss = 0
        for i, (blur_stack, ref_stack) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            blur_stack = blur_stack.squeeze(0).to(device)
            ref_stack = ref_stack.squeeze(0).to(device)

            optimizer.zero_grad()
            restored = model(blur_stack)
            loss = criterion(restored, ref_stack)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch == 0 and i < 3:
                visualize_restoration(blur_stack[0], restored[0], ref_stack[0], index=f"{epoch}_{i}")

        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "C:/Users/IIPL02/Desktop/NEW/KADID_enhance/blur/collaborate_model/weights/collab_unet_blur_ref.pth")
    print("✅ 모델 저장 완료")

if __name__ == "__main__":
    train_collaborative_unet()
