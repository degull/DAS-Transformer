import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LIVEDistortionDatasetRef(Dataset):
    def __init__(self, csv_path, img_root_dir, ref_dir_name="refimgs", transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir
        self.ref_dir = os.path.join(img_root_dir, ref_dir_name)
        self.transform = transform
        self.distortion_classes = ["jp2k", "jpeg", "wn", "gblur", "fastfading"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        distortion_type = row["distortion_type"]
        distorted_filename = row["distortedname"]
        ref_filename = row["refname"]

        # Distorted 이미지 경로
        dist_img_path = os.path.join(self.img_root_dir, distortion_type, distorted_filename)
        # Reference 이미지 경로
        ref_img_path = os.path.join(self.ref_dir, ref_filename)

        # 이미지 로드
        dist_img = Image.open(dist_img_path).convert("RGB")
        ref_img = Image.open(ref_img_path).convert("RGB")

        if self.transform:
            dist_img = self.transform(dist_img)
            ref_img = self.transform(ref_img)

        label = self.distortion_classes.index(distortion_type)

        return dist_img, ref_img, label
