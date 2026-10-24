
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config import DATASET_DIR, IMAGE_SIZE


# ImageNet normalization for pretrained model weights
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class BovineDataset(Dataset):
    def __init__(self, csv_path, transform=None, class_to_idx=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform

        classes = sorted(self.df["breed"].unique())

        if class_to_idx is None:
            self.class_to_idx = {
                breed: idx for idx, breed in enumerate(classes)
            }
        else:
            self.class_to_idx = class_to_idx

        unknown = set(classes) - set(self.class_to_idx)
        if unknown:
            raise ValueError(f"Unknown breed labels: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = DATASET_DIR / str(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["breed"]]
        return image, torch.tensor(label, dtype=torch.long)
