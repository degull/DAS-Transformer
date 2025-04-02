# nafnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# SimpleGate: 채널을 반으로 나눈 뒤 element-wise 곱
def simple_gate(x):
    x1, x2 = x.chunk(2, dim=1)
    return x1 * x2

# Simplified Channel Attention (SCA)
class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        w = self.pool(x)
        w = self.conv(w)
        return x * w

# NAFBlock (Nonlinear Activation Free)
class NAFBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden_channels = channels * expansion

        self.norm1 = nn.LayerNorm([channels, 1, 1])
        self.pwconv1 = nn.Conv2d(channels, hidden_channels * 2, 1)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1, groups=hidden_channels * 2)
        self.pwconv2 = nn.Conv2d(hidden_channels, channels, 1)
        self.sca = SimplifiedChannelAttention(channels)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = simple_gate(x)
        x = self.pwconv2(x)
        x = self.sca(x)
        return x + shortcut

# Downsample (stride-2 convolution)
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        return self.conv(x)

# Upsample (Pixel Shuffle)
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

# 전체 NAFNet
class NAFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=32, num_blocks=[2, 4, 8, 4, 2]):
        super().__init__()

        self.entry = nn.Conv2d(in_channels, width, 3, padding=1)

        # Encoder
        self.enc1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[0])])
        self.down1 = Downsample(width, width * 2)

        self.enc2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks[1])])
        self.down2 = Downsample(width * 2, width * 4)

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(width * 4) for _ in range(num_blocks[2])])

        # Decoder
        self.up2 = Upsample(width * 4, width * 2)
        self.dec2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(num_blocks[3])])

        self.up1 = Upsample(width * 2, width)
        self.dec1 = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks[4])])

        self.exit = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.entry(x)

        enc1 = self.enc1(x)
        x = self.down1(enc1)

        enc2 = self.enc2(x)
        x = self.down2(enc2)

        x = self.middle(x)

        x = self.up2(x) + enc2
        x = self.dec2(x)

        x = self.up1(x) + enc1
        x = self.dec1(x)

        return self.exit(x)

if __name__ == "__main__":
    model = NAFNet(in_channels=3, out_channels=3, width=32)
    x = torch.randn(1, 3, 256, 256)  # 배치 1, RGB 이미지
    out = model(x)
    print(out.shape)  # torch.Size([1, 3, 256, 256])
