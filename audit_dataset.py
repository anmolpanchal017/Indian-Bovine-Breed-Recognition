
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Current source layout. Metadata paths are relative to DATASET_DIR.
DATASET_DIR = Path("Indian_bovine_breeds") / "Indian_bovine_breeds"
CSV_PATH = Path("bovine_breeds_metadata.csv")
REPORT_PATH = Path("reports") / "dataset_audit_report.csv"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH.resolve()}\n"
            "Check your dataset folder and CSV filename."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {"image_id", "breed", "path"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\n" + "=" * 55)
    print("BOVINE BREED DATASET AUDIT")
    print("=" * 55)

    print(f"\nMetadata records: {len(df)}")
    print(f"Unique breeds: {df['breed'].nunique()}")
    print(f"Duplicate paths: {df['path'].duplicated().sum()}")

    print("\nImages per breed:")
    print(df["breed"].value_counts().to_string())

    missing_files = []
    invalid_images = []
    extensions = Counter()
    valid_images = 0

    for row in df.itertuples(index=False):
        relative_path = str(getattr(row, "path"))
        image_path = DATASET_DIR / relative_path

        if not image_path.is_file():
            missing_files.append(relative_path)
            continue

        extensions[image_path.suffix.lower()] += 1

        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images += 1

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ):
            invalid_images.append(relative_path)

    # Verify folder name against breed label
    normalized_paths = (
        df["path"].astype(str).str.replace("\\", "/", regex=False)
    )

    actual_folders = normalized_paths.str.split("/").str[0]

    folder_mismatches = (
        actual_folders.str.casefold()
        != df["breed"].astype(str).str.strip().str.casefold()
    )

    print("\n" + "=" * 55)
    print("IMAGE VALIDATION")
    print("=" * 55)

    print(f"Valid images: {valid_images}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Unreadable images: {len(invalid_images)}")
    print(f"Image formats: {dict(extensions)}")
    print(f"Folder-label mismatches: {folder_mismatches.sum()}")

    if missing_files:
        print("\nFirst 10 missing files:")
        print(*missing_files[:10], sep="\n")

    if invalid_images:
        print("\nFirst 10 unreadable images:")
        print(*invalid_images[:10], sep="\n")

    if folder_mismatches.any():
        print("\nFirst 10 folder-label mismatches:")
        print(
            df.loc[folder_mismatches, ["breed", "path"]]
            .head(10)
            .to_string(index=False)
        )

    # Save audit results
    df["file_exists"] = [
        (DATASET_DIR / str(p)).is_file()
        for p in df["path"]
    ]

    df["audit_status"] = "valid"

    missing_set = set(missing_files)
    invalid_set = set(invalid_images)

    df.loc[
        df["path"].astype(str).isin(missing_set),
        "audit_status"
    ] = "missing"

    df.loc[
        df["path"].astype(str).isin(invalid_set),
        "audit_status"
    ] = "unreadable"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)

    print(f"\nAudit report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
