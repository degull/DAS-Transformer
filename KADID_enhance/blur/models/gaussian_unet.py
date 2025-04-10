# 기존 UNet + HFA 구조 (Gaussian 전용)
import torch
import torch.nn as nn

# ✅ High-Frequency Attention for Gaussian Blur
class HighFrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_highpass = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        high_freq = self.conv_highpass(x) - x
        attn_map = self.sigmoid(high_freq)
        return x + attn_map * high_freq

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

# ✅ Gaussian Blur 복원용 UNet 모델
class GaussianUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 4)

        self.attention = HighFrequencyAttention(base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 1),
            nn.Tanh()  # [-1, 1]
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))  # 학습 가능한 α (초기값 0.1)

    def forward(self, x):
        residual = x
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)

        b = self.attention(b)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        out = (out + 1) / 2  # [0, 1]로 스케일 복원
        return torch.clamp(out + self.alpha * residual, 0.0, 1.0)

# ✅ 실행 테스트
if __name__ == "__main__":
    model = GaussianUNet()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print("✅ Output shape:", out.shape)
