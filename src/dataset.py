import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDemographicDataset(Dataset):
    """
    Dataset for the project's two modalities:
    chest X-ray image + age + sex.

    Sex encoding:
        M -> 1.0
        F -> 0.0
    """

    def __init__(
        self,
        dataframe,
        image_dir="data/images",
        age_mean=None,
        age_std=None,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_dir = image_dir

        if age_mean is None or age_std is None:
            raise ValueError("age_mean and age_std must be supplied from the training split.")

        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if float(age_std) > 0 else 1.0

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        age = (float(row["age"]) - self.age_mean) / self.age_std
        sex = 1.0 if str(row["sex"]).strip().upper() == "M" else 0.0
        label = float(row["label"])

        return {
            "image": image,
            "age": torch.tensor([age], dtype=torch.float32),
            "sex": torch.tensor([sex], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
            "image_name": row["image"],
        }


def training_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def evaluation_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
