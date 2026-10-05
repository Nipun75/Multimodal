import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    ChestXrayDemographicDataset,
    training_transforms,
    evaluation_transforms,
)
from model import MultimodalCardiomegalyModel


SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_DIR = "data/images"
METADATA_PATH = "data/patient_data.csv"
MODEL_PATH = "models/multimodal_best.pth"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_level_split(df):
    """
    70/15/15 patient-level split.
    No patient_id is allowed to appear in more than one split.
    """
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(
        first.split(df, groups=df["patient_id"])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(
        second.split(temp_df, groups=temp_df["patient_id"])
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def make_loaders(train_df, val_df, test_df):
    age_mean = train_df["age"].astype(float).mean()
    age_std = train_df["age"].astype(float).std()

    train_ds = ChestXrayDemographicDataset(
        train_df,
        IMAGE_DIR,
        age_mean,
        age_std,
        transform=training_transforms(),
    )
    val_ds = ChestXrayDemographicDataset(
        val_df,
        IMAGE_DIR,
        age_mean,
        age_std,
        transform=evaluation_transforms(),
    )
    test_ds = ChestXrayDemographicDataset(
        test_df,
        IMAGE_DIR,
        age_mean,
        age_std,
        transform=evaluation_transforms(),
    )

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    )


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        images = batch["image"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, age, sex)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return total_loss / max(total_items, 1)


def main():
    set_seed()

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    train_df, val_df, test_df = patient_level_split(df)

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalCardiomegalyModel(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_items = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images, age, sex)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)

        train_loss = running_loss / max(total_items, 1)
        val_loss = validation_loss(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "age_mean": float(train_df["age"].astype(float).mean()),
                    "age_std": float(train_df["age"].astype(float).std()),
                },
                MODEL_PATH,
            )
            print(f"Saved best model to {MODEL_PATH}")

    print("Training complete.")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()
