# ASTCompressionRestoration 모델 및 AFRModule 정의

# ast_compression_model.py
""" import torch
import torch.nn as nn
import torch.nn.functional as F


# ✅ Feature Refinement Feed-forward Network (FRFN)
class FRFN(nn.Module):
    def __init__(self, channels):
        super(FRFN, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.pconv = nn.Conv2d(channels, channels, kernel_size=1)
        self.linear1 = nn.Linear(channels, channels)
        self.dwconv = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, groups=channels // 2)
        self.linear2 = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape

        residual = x  # Skip용


        # ✅ 올바른 차원 정렬 적용 (HWxC → BxCxHW)
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        x = self.norm(x)
        x = self.pconv(x.permute(0, 2, 1).view(B, C, H, W))  # (B, C, H, W)

        x = self.linear1(x.view(B, C, H * W).permute(0, 2, 1))  # (B, HW, C)
        x = x.view(B, H, W, C)  # Reshape

        # ✅ Split and Depth-wise Convolution
        x1, x2 = x.split(C // 2, dim=-1)
        x1 = self.dwconv(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (B, H, W, C//2)
        x = torch.cat([x1, x2], dim=-1)  # (B, H, W, C)

        # ✅ 마지막 차원 변환을 고려한 Linear 적용
        x = self.linear2(x.view(B, H * W, C)).view(B, H, W, C)  # (B, H, W, C)

        return x.permute(0, 3, 1, 2) + residual  # 정확한 skip 연결


# ✅ Adaptive Sparse Self-Attention (ASSA)
class ASSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(ASSA, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn1 = self.softmax(Q @ K.transpose(-2, -1))  # Standard attention
        attn2 = torch.relu(Q @ K.transpose(-2, -1)) ** 2  # Sparse attention

        # Adaptive weighting
        w1 = torch.exp(self.w1) / (torch.exp(self.w1) + torch.exp(self.w2))
        w2 = torch.exp(self.w2) / (torch.exp(self.w1) + torch.exp(self.w2))

        attn = w1 * attn1 + w2 * attn2
        output = (attn @ V).transpose(1, 2).view(B, C, H, W)
        return output + x.permute(0, 2, 1).view(B, C, H, W)  # Skip Connection


# ✅ Adaptive Sparse Transformer (AST) 기반 Compression 복원 모델
class ASTCompressionRestoration(nn.Module):
    def __init__(self, img_channels=3, embed_dim=64):
        super(ASTCompressionRestoration, self).__init__()

        # ✅ Encoder
        self.conv1 = nn.Conv2d(img_channels, embed_dim, kernel_size=3, padding=1)
        self.frfn1 = FRFN(embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1)
        self.frfn2 = FRFN(embed_dim * 2)

        # ✅ Transformer Bottleneck (ASSA 적용)
        self.assa1 = ASSA(embed_dim * 2)
        self.frfn3 = FRFN(embed_dim * 2)

        # ✅ Decoder (ASSA + FRFN 적용)
        self.assa2 = ASSA(embed_dim * 2)
        self.frfn4 = FRFN(embed_dim * 2)
        self.deconv1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.frfn5 = FRFN(embed_dim)
        self.deconv2 = nn.Conv2d(embed_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x  # 🔸 입력 이미지 저장 (Residual connection용)

        x1 = F.relu(self.conv1(x))
        x1 = self.frfn1(x1)

        x2 = F.relu(self.conv2(x1))
        x2 = self.frfn2(x2)

        x3 = self.assa1(x2)
        x3 = self.frfn3(x3)

        x4 = self.assa2(x3)
        x4 = self.frfn4(x4)

        x5 = F.relu(self.deconv1(x4))
        x5 = self.frfn5(x5)

        restored = self.deconv2(x5)

        # 🔸 Residual 방식으로 입력 이미지 더함
        restored = restored + x  # x는 입력 이미지

        # 🔸 출력 범위를 clamp해서 색상 깨짐 방지
        restored = torch.clamp(restored, 0.0, 1.0)

        return restored




# ✅ 실행 테스트 코드 추가
if __name__ == "__main__":
    print("🔹 ASTCompressionRestoration 모델 테스트 중...")

    # 모델 초기화
    model = ASTCompressionRestoration(img_channels=3, embed_dim=64)
    model.eval()

    # 임의의 입력 이미지 생성 (batch_size=1, C=3, H=256, W=256)
    sample_input = torch.randn(1, 3, 256, 256)

    # 모델 실행
    with torch.no_grad():
        output = model(sample_input)

    print("✅ 모델 테스트 완료!")
    print(f"입력 크기: {sample_input.shape}")
    print(f"출력 크기: {output.shape}")

 """
# 브랜치 추가

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Feature Refinement Feed-forward Network (FRFN)
class FRFN(nn.Module):
    def __init__(self, channels):
        super(FRFN, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.pconv = nn.Conv2d(channels, channels, kernel_size=1)
        self.linear1 = nn.Linear(channels, channels)
        self.dwconv = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, groups=channels // 2)
        self.linear2 = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.norm(x)
        x = self.pconv(x.permute(0, 2, 1).view(B, C, H, W))
        x = self.linear1(x.view(B, C, H * W).permute(0, 2, 1))
        x = x.view(B, H, W, C)
        x1, x2 = x.split(C // 2, dim=-1)
        x1 = self.dwconv(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.cat([x1, x2], dim=-1)
        x = self.linear2(x.view(B, H * W, C)).view(B, H, W, C)
        return x.permute(0, 3, 1, 2) + residual

# ✅ Adaptive Sparse Self-Attention (ASSA)
class ASSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(ASSA, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = self.norm(x.flatten(2).transpose(1, 2))
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        attn1 = self.softmax(Q @ K.transpose(-2, -1))
        attn2 = torch.relu(Q @ K.transpose(-2, -1)) ** 2
        w1 = torch.exp(self.w1) / (torch.exp(self.w1) + torch.exp(self.w2))
        w2 = torch.exp(self.w2) / (torch.exp(self.w1) + torch.exp(self.w2))
        attn = w1 * attn1 + w2 * attn2
        output = (attn @ V).transpose(1, 2).view(B, C, H, W)
        return output + x

# ✅ JPEG: DilatedConv + Edge-Aware Block
class EdgeAwareBlock(nn.Module):
    def __init__(self, channels):
        super(EdgeAwareBlock, self).__init__()
        self.dilated = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        edge_feat = self.dilated(x)
        edge_map = torch.sigmoid(self.edge_conv(edge_feat))
        return x + edge_feat * edge_map

# ✅ JPEG2000: Non-local Attention + Texture Refiner
class TextureRefiner(nn.Module):
    def __init__(self, channels):
        super(TextureRefiner, self).__init__()
        self.non_local = nn.Conv2d(channels, channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        attn = torch.sigmoid(self.non_local(x))
        texture = self.refine(x)
        return x + attn * texture

# ✅ AST Compression Restoration
class ASTCompressionRestoration(nn.Module):
    def __init__(self, img_channels=3, embed_dim=64):
        super(ASTCompressionRestoration, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, embed_dim, 3, padding=1)
        self.frfn1 = FRFN(embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim * 2, 3, stride=2, padding=1)
        self.frfn2 = FRFN(embed_dim * 2)
        self.assa1 = ASSA(embed_dim * 2)
        self.frfn3 = FRFN(embed_dim * 2)
        self.assa2 = ASSA(embed_dim * 2)
        self.frfn4 = FRFN(embed_dim * 2)
        self.deconv1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, 3, stride=2, padding=1, output_padding=1)
        self.frfn5 = FRFN(embed_dim)

        # 브랜치 정의
        self.jpeg_branch = nn.Sequential(
            EdgeAwareBlock(embed_dim),
            nn.Conv2d(embed_dim, img_channels, 3, padding=1)
        )
        self.jpeg2000_branch = nn.Sequential(
            TextureRefiner(embed_dim),
            nn.Conv2d(embed_dim, img_channels, 3, padding=1)
        )
        self.default_branch = nn.Conv2d(embed_dim, img_channels, 3, padding=1)

    def forward(self, x, class_id=None):
        x1 = F.relu(self.conv1(x))
        x1 = self.frfn1(x1)
        x2 = F.relu(self.conv2(x1))
        x2 = self.frfn2(x2)
        x3 = self.assa1(x2)
        x3 = self.frfn3(x3)
        x4 = self.assa2(x3)
        x4 = self.frfn4(x4)
        x5 = F.relu(self.deconv1(x4))
        x5 = self.frfn5(x5)

        # class_id 분기 처리
        if class_id is not None:
            outputs = []
            if isinstance(class_id, int):
                class_id = torch.tensor([class_id] * x.size(0), device=x.device)
            for i in range(x.size(0)):
                if class_id[i] == 10:
                    print("📌 JPEG 경계 복원 브랜치 적용")
                    out = self.jpeg_branch(x5[i].unsqueeze(0))
                elif class_id[i] == 9:
                    print("📌 JPEG2000 텍스처 복원 브랜치 적용")
                    out = self.jpeg2000_branch(x5[i].unsqueeze(0))
                else:
                    print("⚠️ Unknown class_id, 일반 복원 사용")
                    out = self.default_branch(x5[i].unsqueeze(0))
                outputs.append(out)
            restored = torch.cat(outputs, dim=0)
        else:
            print("⚠️ class_id 미제공. 일반 복원 브랜치 사용.")
            restored = self.default_branch(x5)

        restored = restored + x
        restored = torch.clamp(restored, 0.0, 1.0)
        return restored

# ✅ 실행 테스트 코드
if __name__ == "__main__":
    print("🔹 ASTCompressionRestoration 모델 테스트 중...\n")

    model = ASTCompressionRestoration()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)

    print("[TEST] JPEG (class_id=10)")
    out_jpeg = model(input_tensor, class_id=10)
    print("🔸 JPEG 출력 크기:", out_jpeg.shape)

    print("\n[TEST] JPEG2000 (class_id=9)")
    out_jpeg2000 = model(input_tensor, class_id=9)
    print("🔸 JPEG2000 출력 크기:", out_jpeg2000.shape)

    print("\n[TEST] 일반 (class_id=None)")
    out_default = model(input_tensor)
    print("🔸 기본 출력 크기:", out_default.shape)
