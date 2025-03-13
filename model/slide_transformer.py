import torch
import torch.nn as nn
from timm.layers import trunc_normal_

class SlideAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dep_conv = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=kernel_size, bias=True,
                                  groups=self.head_dim, padding=kernel_size // 2)

        self.relative_position_bias_table = None

    def forward(self, x):
        B, N, C = x.shape  
        H = W = int(N ** 0.5)  

        if self.relative_position_bias_table is None or self.relative_position_bias_table.shape[-1] != H * W:
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.num_heads, H * W, H * W))
            trunc_normal_(self.relative_position_bias_table, std=.02)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)
        k = self.dep_conv(k)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)
        v = self.dep_conv(v)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ✅ self.relative_position_bias_table을 `attn`의 device로 이동
        attn = attn + self.relative_position_bias_table.to(attn.device)[:, :, :N, :N]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SlideTransformer(nn.Module):
    """ ✅ DAS-Transformer (CNN + Transformer 결합) """
    def __init__(self, img_size=224, num_classes=6, embed_dim=96, num_heads=6, kernel_size=3, mlp_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ✅ CNN 적용 후 Feature Extraction
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ✅ Slide Attention 적용
        self.attn_layer = SlideAttention(dim=embed_dim, num_heads=num_heads, kernel_size=kernel_size)

        # ✅ MLP 및 Transformer 적용
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.global_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ✅ CNN 출력 크기 변환 (B, C, H*W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # ✅ Slide Attention 적용
        x = self.attn_layer(x)

        # ✅ MLP 적용
        x = self.mlp(x.mean(dim=1))

        # ✅ Global Transformer 적용
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)

        return self.head(x)

# ✅ 테스트 실행
if __name__ == "__main__":
    model = SlideTransformer(img_size=224, num_classes=6)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output Shape:", output.shape)  # ✅ 예상: torch.Size([1, 6])
