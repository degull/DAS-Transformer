import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


# 1️⃣ Collaborative UNet 구성
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


# 2️⃣ Patch 생성 함수
def extract_patches_from_image(image: torch.Tensor, patch_size=128, stride=128):
    _, H, W = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    nH, nW = patches.shape[1], patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
    return patches


# 3️⃣ 비교 시각화 함수
def visualize_comparison(original, restored, index, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(to_pil(original.cpu()))
    axes[0].set_title("Original Blur")
    axes[0].axis("off")
    axes[1].imshow(to_pil(restored.cpu()))
    axes[1].set_title("Restored")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_{index}.png"))
    plt.close()


# 4️⃣ 전체 실행 함수
def run_kadid_collab_deblurring(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    blur_codes = ['01', '02', '03']
    transform = transforms.ToTensor()
    model = CollaborativeUNet().eval()
    count = 0

    for _, row in df.iterrows():
        filename = row['dist_img']
        if any(code in filename for code in blur_codes):
            img_path = os.path.join(img_dir, filename)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            patches = extract_patches_from_image(img_tensor, patch_size=128, stride=128)
            if patches.shape[0] < 1:
                continue
            with torch.no_grad():
                restored_patches = model(patches)
                visualize_comparison(patches[0], restored_patches[0].clamp(0, 1), count)
                count += 1
            if count >= 5:
                break


# 5️⃣ 경로 설정 후 실행
if __name__ == "__main__":
    run_kadid_collab_deblurring(
        csv_path="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/kadid10k.csv",
        img_dir="C:/Users/IIPL02/Desktop/NEW/data/KADID10K/images",
    )

