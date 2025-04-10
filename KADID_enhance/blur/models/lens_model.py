# Lens Blur 전용 구조 (CVPR 2024 논문 기반)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 위치 정보 기반 Conv로 position encoding 생성
class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = torch.linspace(-1, 1, W, device=x.device)
        ys = torch.linspace(-1, 1, H, device=x.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        pos = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        pos_feat = self.conv(pos)
        return pos_feat

# ✅ Low-Rank Projection 모듈
class LowRankAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)        # [B, HW, C]
        attn = self.proj(x_flat).permute(0, 2, 1).view(B, C, H, W)
        attn = self.sigmoid(attn)
        return x * attn + x

# ✅ 기본 UNet 블록
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# ✅ Lens Blur 복원 모델 (Position-aware + Low-Rank Attention)
class LensDeblurUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 4)

        self.pos_enc = PositionalEncoding(base_channels * 4)
        self.lowrank_attn = LowRankAttention(base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 1),
            nn.Tanh()
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))  # 학습 가능한 잔차 비율

    def forward(self, x):
        residual = x
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)

        pos_feat = self.pos_enc(b)
        b = b + pos_feat
        b = self.lowrank_attn(b)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        out = (out + 1) / 2  # [0, 1]로 정규화
        return torch.clamp(out + self.alpha * residual, 0.0, 1.0)

# ✅ 실행 테스트
if __name__ == "__main__":
    model = LensDeblurUNet()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("✅ Output shape:", out.shape)
