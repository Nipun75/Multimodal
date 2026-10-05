from pathlib import Path
import shutil
import pandas as pd

# Kaggle NIH Chest X-ray dataset preparation.
# Target: 0 = No Finding, 1 = Cardiomegaly.
# Only exact single-label cases are retained; multi-label cases are excluded.

DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "Data_Entry_2017.csv"
OUTPUT_IMAGE_DIR = DATA_DIR / "images"
OUTPUT_CSV = DATA_DIR / "patient_data.csv"

MAX_SAMPLES = 1000
BALANCE_CLASSES = True
RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
]


def find_image_files(data_dir: Path) -> dict[str, Path]:
    """Index NIH PNG files once across images_001, images_002, etc."""
    image_files = {}
    image_dirs = sorted(data_dir.glob("images_*"))

    if not image_dirs:
        raise FileNotFoundError(
            f"No image folders found under {data_dir}. "
            "Expected images_001, images_002, ... images_012."
        )

    print(f"Found {len(image_dirs)} image folders.")

    for image_dir in image_dirs:
        if image_dir.is_dir():
            for image_path in image_dir.glob("*.png"):
                image_files[image_path.name] = image_path

    if not image_files:
        raise FileNotFoundError(
            f"No PNG images found under {data_dir}/images_*/."
        )

    print(f"Indexed {len(image_files):,} image files.")
    return image_files


def clean_age(value):
    """Convert Patient Age to a valid integer age."""
    age = pd.to_numeric(value, errors="coerce")

    if pd.isna(age):
        return None

    age = float(age)

    if age < 0 or age > 120:
        return None

    return int(age)


def main():
    print("=" * 60)
    print("NIH CHEST X-RAY DATASET PREPARATION")
    print("=" * 60)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}\n"
            "Place Kaggle's Data_Entry_2017.csv inside data/."
        )

    # 1. Load Kaggle metadata.
    print(f"\nLoading metadata: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)

    print(f"Total metadata rows: {len(df):,}")
    print("Columns:")
    print(df.columns.tolist())

    missing_columns = [
        c for c in REQUIRED_COLUMNS if c not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    # 2. Exact target filtering.
    # Multi-label rows such as "Cardiomegaly|Effusion" are excluded.
    target_labels = {"No Finding": 0, "Cardiomegaly": 1}

    filtered = df[df["Finding Labels"].isin(target_labels)].copy()
    filtered["label"] = filtered["Finding Labels"].map(target_labels)
    filtered["disease"] = filtered["Finding Labels"]

    print(
        f"\nExact No Finding/Cardiomegaly rows: "
        f"{len(filtered):,}"
    )

    # 3. Clean demographics.
    filtered["age"] = filtered["Patient Age"].apply(clean_age)
    filtered["sex"] = (
        filtered["Patient Gender"].astype(str).str.strip().str.upper()
    )

    filtered = filtered[filtered["sex"].isin(["M", "F"])].copy()
    filtered = filtered[filtered["age"].notna()].copy()
    filtered["age"] = filtered["age"].astype(int)

    if filtered.empty:
        raise RuntimeError("No usable rows remain after age/sex filtering.")

    print(f"Rows after age/sex validation: {len(filtered):,}")

    # 4. Locate actual X-ray files.
    image_index = find_image_files(DATA_DIR)
    filtered["image_path"] = filtered["Image Index"].map(image_index)

    missing_images = filtered["image_path"].isna()
    if missing_images.any():
        print(
            f"Warning: {int(missing_images.sum()):,} selected rows have "
            "no matching image and will be removed."
        )

    filtered = filtered[~missing_images].copy()

    if filtered.empty:
        raise RuntimeError(
            "No selected rows have matching images. "
            "Check the Kaggle image folders under data/."
        )

    # 5. Select an initial balanced sample.
    if BALANCE_CLASSES:
        counts = filtered["label"].value_counts()

        if len(counts) < 2:
            raise RuntimeError(
                "Both No Finding and Cardiomegaly are required."
            )

        per_class = min(counts.min(), MAX_SAMPLES // 2)

        if per_class == 0:
            raise RuntimeError("MAX_SAMPLES is too small.")

        parts = []
        for label in [0, 1]:
            class_df = filtered[filtered["label"] == label]
            parts.append(
                class_df.sample(
                    n=per_class,
                    random_state=RANDOM_SEED,
                )
            )

        filtered = pd.concat(parts, ignore_index=True)
    elif len(filtered) > MAX_SAMPLES:
        filtered = filtered.sample(
            n=MAX_SAMPLES,
            random_state=RANDOM_SEED,
        )

    filtered = filtered.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print(f"\nFinal selected samples: {len(filtered):,}")

    # 6. Copy selected images using the original NIH filenames.
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\nCopying selected X-ray images...")

    for i, (_, row) in enumerate(filtered.iterrows(), start=1):
        source = Path(row["image_path"])
        destination = OUTPUT_IMAGE_DIR / source.name

        if not destination.exists():
            shutil.copy2(source, destination)

        rows.append(
            {
                "image": source.name,
                "patient_id": str(row["Patient ID"]),
                "age": int(row["age"]),
                "sex": row["sex"],
                "label": int(row["label"]),
                "disease": row["disease"],
            }
        )

        if i % 100 == 0 or i == len(filtered):
            print(f"  {i:,}/{len(filtered):,} images prepared")

    # 7. Save compact metadata.
    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    # 8. Sanity checks.
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Metadata : {OUTPUT_CSV}")
    print(f"Images   : {OUTPUT_IMAGE_DIR}")
    print(f"Shape    : {output_df.shape}")

    print("\nClass distribution:")
    print(output_df["disease"].value_counts())

    print("\nLabel distribution:")
    print(output_df["label"].value_counts().sort_index())

    print("\nSex distribution:")
    print(output_df["sex"].value_counts())

    print("\nAge statistics:")
    print(output_df["age"].describe())

    print(f"\nUnique patients: {output_df['patient_id'].nunique():,}")

    print("\nFirst 5 rows:")
    print(output_df.head())

    print(
        "\nNext step: create a patient-level 70/15/15 "
        "train/validation/test split using patient_id."
    )


if __name__ == "__main__":
    main()
