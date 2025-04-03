import torch
import torch.nn as nn
import torch.nn.functional as F


# ✅ Stack-wise Feature Pooling
class StackFeaturePooling(nn.Module):
    def __init__(self, pooling_type='avg'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, feat_stack):  # [N, C, H, W]
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
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ✅ Full UNet with Collaborative Pooling
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
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetConvBlock(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, in_channels, 1)

        self.pooling = StackFeaturePooling(pooling_type)
        self.fuse = nn.Conv2d(base_channels * 2, base_channels, 1)

    def forward(self, x_stack):  # [N, C, H, W]
        e1 = self.enc1(x_stack)  # [N, 64, H, W]
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)       # [N, 128, H/2, W/2]
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)  # [N, 256, H/4, W/4]

        d2 = self.up2(b)         # [N, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # skip connection
        d2 = self.dec2(d2)

        pooled = self.pooling(d2)  # collaborative pooling
        d2 = self.fuse(torch.cat([d2, pooled], dim=1))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)  # [N, C, H, W]
        return out


# ✅ 예시 실행
if __name__ == "__main__":
    model = CollaborativeUNet(pooling_type='avg')
    dummy_input = torch.randn(4, 3, 128, 128)  # 4 patches from same blur image
    restored = model(dummy_input)
    print("Restored shape:", restored.shape)  # → [4, 3, 128, 128]
