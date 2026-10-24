
from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
METADATA_PATH = (
    ROOT / "data" / "processed"
    / "metadata_without_conflicting_duplicates.csv"
)
DATASET_DIR = ROOT / "Indian_bovine_breeds" / "Indian_bovine_breeds"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "reports"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(METADATA_PATH).copy()
df["path"] = df["path"].astype(str)

# Assign a content hash so exact copies stay in the same split.
hashes = []

for relative_path in df["path"]:
    image_path = DATASET_DIR / relative_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    hasher = hashlib.sha256()
    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    hashes.append(hasher.hexdigest())

df["content_hash"] = hashes

# Each content hash must have exactly one breed label.
labels_per_hash = df.groupby("content_hash")["breed"].nunique()
if (labels_per_hash > 1).any():
    raise ValueError(
        "Conflicting labels remain. Resolve them before splitting."
    )

# Create one row per unique image content for splitting.
groups = (
    df[["content_hash", "breed"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# First split: 70% train, 30% temporary.
train_groups, temp_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=42,
    stratify=groups["breed"],
)

# Second split: divide temporary data equally into val and test.
val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=42,
    stratify=temp_groups["breed"],
)

split_by_hash = {}

for hashes_part, split_name in [
    (train_groups, "train"),
    (val_groups, "val"),
    (test_groups, "test"),
]:
    for content_hash in hashes_part["content_hash"]:
        split_by_hash[content_hash] = split_name

df["split"] = df["content_hash"].map(split_by_hash)

# Save all records, retaining same-breed exact copies together.
for split_name in ["train", "val", "test"]:
    part = df[df["split"] == split_name].drop(
        columns=["content_hash", "split"]
    )
    part.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False)

df.drop(columns=["content_hash"]).to_csv(
    SPLIT_DIR / "all_splits.csv", index=False
)

# Report actual record counts and unique image counts.
summary = (
    df.groupby(["split", "breed"])
    .agg(
        records=("path", "size"),
        unique_images=("content_hash", "nunique"),
    )
    .reset_index()
)
summary.to_csv(REPORT_DIR / "split_summary_by_breed.csv", index=False)

print("=" * 60)
print("DATASET SPLIT SUMMARY")
print("=" * 60)

for split_name in ["train", "val", "test"]:
    part = df[df["split"] == split_name]
    print(
        f"{split_name.upper():5s}: "
        f"{len(part)} records, "
        f"{part['content_hash'].nunique()} unique images, "
        f"{part['breed'].nunique()} breeds"
    )

# Verify that no image content crosses splits.
splits_per_hash = df.groupby("content_hash")["split"].nunique()
assert splits_per_hash.max() == 1, "Duplicate leakage detected!"

# Verify that every breed appears in each split.
breed_sets = [
    set(df.loc[df["split"] == name, "breed"])
    for name in ["train", "val", "test"]
]
assert all(breed_sets), "One or more splits are empty!"
assert all(breed_sets[0] == s for s in breed_sets[1:]), (
    "Some breeds are missing from a split."
)

print("\nDuplicate leakage check: PASSED")
print("Breed representation check: PASSED")
print(f"Split files saved in: {SPLIT_DIR}")
print(f"Per-breed report saved in: {REPORT_DIR}")
