import torch
import torch.nn as nn
from model.slide_transformer import SlideTransformer  # 기존 backbone 사용

class SlideTransformerRef(nn.Module):
    def __init__(self, img_size=224, num_classes=5, embed_dim=96):
        super(SlideTransformerRef, self).__init__()
        self.backbone = SlideTransformer(img_size=img_size, num_classes=embed_dim)  # ✅ num_classes = embed_dim로 세팅

        # 🔥 두 feature의 차이만 사용 → 입력 차원 = embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, dist_img, ref_img):
        # backbone의 출력은 self.head(x) → 즉, feature dim = embed_dim
        feat_dist = self.backbone(dist_img, mode="train")  # [B, embed_dim]
        feat_ref = self.backbone(ref_img, mode="train")    # [B, embed_dim]

        feat_diff = torch.abs(feat_dist - feat_ref)  # 🔥 핵심: 절댓값 차이 사용

        out = self.classifier(feat_diff)  # 분류 결과
        return out
