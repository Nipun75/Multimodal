from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

ds = load_dataset(
    "chehablab/NIHChestXR",
    split="train"
)

print("Total dataset size:", len(ds))


# ==========================================
# 2. SELECT ONLY REQUIRED CLASSES
# ==========================================

# Label mapping:
# 0 = No Finding
# 2 = Cardiomegaly

print("\nSelecting Cardiomegaly and No Finding...")


def select_required_labels(example):

    labels = example["labels"]

    # Keep only images having exactly one label
    # This avoids samples having multiple diseases.
    if len(labels) != 1:
        return False

    return labels[0] in [0, 2]


filtered = ds.filter(select_required_labels)

print("Filtered dataset size:", len(filtered))


# ==========================================
# 3. LIMIT DATASET SIZE
# ==========================================

# Change these numbers if required.
# We start small for partial execution.

MAX_SAMPLES = 1000

if len(filtered) > MAX_SAMPLES:
    filtered = filtered.shuffle(seed=42).select(
        range(MAX_SAMPLES)
    )

print("Samples selected:", len(filtered))


# ==========================================
# 4. CREATE DATA FOLDER
# ==========================================

os.makedirs("data/images", exist_ok=True)


# ==========================================
# 5. SAVE IMAGES + METADATA
# ==========================================

metadata = []

print("\nSaving images...")

for i, sample in enumerate(filtered):

    # Get image
    image = sample["image"]

    # Convert to RGB
    image = image.convert("RGB")

    # Create filename
    filename = f"image_{i:04d}.jpg"

    image_path = os.path.join(
        "data",
        "images",
        filename
    )

    # Save image
    image.save(image_path)

    # Original label
    original_label = sample["labels"][0]

    # Convert to binary target
    if original_label == 2:
        disease = 1
        disease_name = "Cardiomegaly"
    else:
        disease = 0
        disease_name = "No Finding"

    # Save metadata
    metadata.append({
        "image": filename,
        "patient_id": sample["patient_id"],
        "scan_id": sample["scan_id"],
        "age": sample["age"],
        "sex": sample["sex"],
        "label": disease,
        "disease": disease_name
    })

    if (i + 1) % 100 == 0:
        print(f"{i + 1} images saved...")


# ==========================================
# 6. CREATE CSV
# ==========================================

df = pd.DataFrame(metadata)

df.to_csv(
    "data/patient_data.csv",
    index=False
)


# ==========================================
# 7. DISPLAY INFORMATION
# ==========================================

print("\n====================================")
print("DATASET PREPARATION COMPLETED")
print("====================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["disease"].value_counts())

print("\nSex distribution:")
print(df["sex"].value_counts())

print("\nAge statistics:")
print(df["age"].describe())

print("\nSaved files:")
print("Images  : data/images/")
print("Metadata: data/patient_data.csv")