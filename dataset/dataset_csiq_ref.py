import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CSIQDatasetFR(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.dist_dir = os.path.join(root_dir, "dst_imgs")
        self.src_dir = os.path.join(root_dir, "src_imgs")
        self.transform = transform

        self.distortion_types = ["AWGN", "BLUR", "contrast", "fnoise", "JPEG", "jpeg2000"]
        self.image_list = []

        for label, distortion in enumerate(self.distortion_types):
            dist_folder = os.path.join(self.dist_dir, distortion)
            for img_name in os.listdir(dist_folder):
                # ✅ 예: 'bridge.BLUR.3.png' → 'bridge.png'
                base_name = img_name.split(".")[0] + ".png"
                ref_path = os.path.join(self.src_dir, base_name)

                self.image_list.append({
                    "dist_path": os.path.join(dist_folder, img_name),
                    "ref_path": ref_path,
                    "label": label
                })


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]
        dist_img = Image.open(item["dist_path"]).convert("RGB")
        ref_img = Image.open(item["ref_path"]).convert("RGB")

        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        return ref_img, dist_img, item["label"]
