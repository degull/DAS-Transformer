# collaborate_blur_model.py
""" import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Gaussian: High-Frequency Attention
class HighFrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_highpass = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        high_freq = self.conv_highpass(x) - x
        attn_map = self.sigmoid(high_freq)
        return x + attn_map * high_freq

# ✅ Lens Blur: Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        attn = torch.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))
        return x * attn + x

# ✅ Motion Blur: Directional Attention
class DirectionalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0))
        self.gate = nn.Sigmoid()

    def forward(self, x):
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)
        attn = self.gate(h_feat + v_feat)
        return x + x * attn

# ✅ 기본 UNet 블록
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

# ✅ UNet + Attention 모듈
class BlurUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, attention_module=None):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.attention = attention_module(base_channels) if attention_module else None

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        if self.attention:
            b = self.attention(b)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# ✅ 블러별 브랜치 포함 전체 모델
class BlurBranchRestorationModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.gaussian_branch = BlurUNet(in_channels, base_channels, attention_module=HighFrequencyAttention)
        self.lens_branch = BlurUNet(in_channels, base_channels, attention_module=SpatialAttention)
        self.motion_branch = BlurUNet(in_channels, base_channels, attention_module=DirectionalAttention)
        self.default_branch = BlurUNet(in_channels, base_channels)

    def forward(self, x, class_id=None):
        outputs = []
        for i in range(x.size(0)):
            cid = class_id[i].item() if class_id is not None else -1
            if cid == 1:
                print("📌 Gaussian Blur 복원 (고주파 텍스처)")
                out = self.gaussian_branch(x[i].unsqueeze(0))
            elif cid == 2:
                print("📌 Lens Blur 복원 (공간 가변 흐림)")
                out = self.lens_branch(x[i].unsqueeze(0))
            elif cid == 3:
                print("📌 Motion Blur 복원 (방향성 흐림)")
                out = self.motion_branch(x[i].unsqueeze(0))
            else:
                print("⚠️ Unknown class, 기본 복원 사용")
                out = self.default_branch(x[i].unsqueeze(0))
            outputs.append(out)
        return torch.cat(outputs, dim=0)

# ✅ 실행 테스트 코드
if __name__ == "__main__":
    print("🔹 BlurBranchRestorationModel 테스트 중...\n")

    model = BlurBranchRestorationModel()
    model.eval()
    input_tensor = torch.randn(4, 3, 256, 256)
    class_ids = torch.tensor([1, 2, 3, 1])  # Gaussian, Lens, Motion, Gaussian

    output = model(input_tensor, class_ids)
    print("🔸 최종 출력 크기:", output.shape)  # [4, 3, 256, 256]
 """

# 논문 기반
# collaborate_blur_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Gaussian: High-Frequency Attention
class HighFrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_highpass = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        high_freq = self.conv_highpass(x) - x
        attn_map = self.sigmoid(high_freq)
        return x + attn_map * high_freq

# ✅ Lens Blur: Position-aware Low-rank Attention
class PositionAwareLowRankAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pos_conv = nn.Conv2d(2, channels, 1)
        self.lowrank_proj = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        xs = torch.linspace(-1, 1, W, device=x.device)
        ys = torch.linspace(-1, 1, H, device=x.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        pos = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        pos_feat = self.pos_conv(pos)
        fused = x + pos_feat
        fused_flat = fused.flatten(2).permute(0, 2, 1)
        attn = self.lowrank_proj(fused_flat).permute(0, 2, 1).view(B, C, H, W)
        attn = self.sigmoid(attn)
        return x * attn + x

# ✅ Motion Blur: Blur Pixel Discretization Attention
class BlurPixelDiscretizationAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.level_embed = nn.Embedding(4, channels)

    def forward(self, x):
        blur_level = torch.clamp((x.mean(dim=1, keepdim=True) * 4).long(), 0, 3)
        embed = self.level_embed(blur_level.squeeze(1))
        embed = embed.permute(0, 3, 1, 2)
        x_attended = self.conv(x) + embed
        return x + x_attended

# ✅ 기본 UNet 블록
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

# ✅ UNet + Attention 모듈
class BlurUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, attention_module=None):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = UNetBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = UNetBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.attention = attention_module(base_channels) if attention_module else None

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        if self.attention:
            b = self.attention(b)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# ✅ 블러별 브랜치 포함 전체 모델
class BlurBranchRestorationModel(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.gaussian_branch = BlurUNet(in_channels, base_channels, attention_module=HighFrequencyAttention)
        self.lens_branch = BlurUNet(in_channels, base_channels, attention_module=PositionAwareLowRankAttention)
        self.motion_branch = BlurUNet(in_channels, base_channels, attention_module=BlurPixelDiscretizationAttention)
        self.default_branch = BlurUNet(in_channels, base_channels)

    def forward(self, x, class_id=None):
        outputs = []
        for i in range(x.size(0)):
            cid = class_id[i].item() if class_id is not None else -1
            if cid == 1:
                print("📌 Gaussian Blur 복원 (고주파 텍스처)")
                out = self.gaussian_branch(x[i].unsqueeze(0))
            elif cid == 2:
                print("📌 Lens Blur 복원 (위치 기반 Low-Rank)")
                out = self.lens_branch(x[i].unsqueeze(0))
            elif cid == 3:
                print("📌 Motion Blur 복원 (픽셀 블러 수준 기반)")
                out = self.motion_branch(x[i].unsqueeze(0))
            else:
                print("⚠️ Unknown class, 기본 복원 사용")
                out = self.default_branch(x[i].unsqueeze(0))
            outputs.append(out)
        return torch.cat(outputs, dim=0)

# ✅ 실행 테스트 코드
if __name__ == "__main__":
    print("🔹 BlurBranchRestorationModel 테스트 중...\n")

    model = BlurBranchRestorationModel()
    model.eval()
    input_tensor = torch.randn(4, 3, 256, 256)
    class_ids = torch.tensor([1, 2, 3, 1])  # Gaussian, Lens, Motion, Gaussian

    output = model(input_tensor, class_ids)
    print("🔸 최종 출력 크기:", output.shape)  # [4, 3, 256, 256]
