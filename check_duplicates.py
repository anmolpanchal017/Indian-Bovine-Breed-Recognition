
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "bovine_breeds_metadata.csv"
DATASET_DIR = (
    PROJECT_ROOT
    / "Indian_bovine_breeds"
    / "Indian_bovine_breeds"
)
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("EXACT DUPLICATE IMAGE DETECTION")
print("=" * 60)
print(f"Total metadata records: {len(df)}")

# Calculate SHA-256 hash of each file
hash_to_files = defaultdict(list)
missing_files = []

for index, row in df.iterrows():
    image_path = DATASET_DIR / str(row["path"])

    if not image_path.is_file():
        missing_files.append(str(image_path))
        continue

    try:
        hasher = hashlib.sha256()

        with image_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)

        file_hash = hasher.hexdigest()
        hash_to_files[file_hash].append({
            "image_id": row["image_id"],
            "breed": row["breed"],
            "path": str(row["path"]),
        })

    except OSError as error:
        print(f"Could not read {image_path}: {error}")

# Save duplicate groups
duplicate_groups = []

for file_hash, files in hash_to_files.items():
    if len(files) > 1:
        for item in files:
            duplicate_groups.append({
                "sha256": file_hash,
                "duplicate_count": len(files),
                **item,
            })

duplicates_path = REPORT_DIR / "exact_duplicates.csv"
pd.DataFrame(
    duplicate_groups,
    columns=[
        "sha256",
        "duplicate_count",
        "image_id",
        "breed",
        "path",
    ],
).to_csv(duplicates_path, index=False)

# Print summary
print(f"Files successfully scanned: {sum(map(len, hash_to_files.values()))}")
print(f"Missing files: {len(missing_files)}")
print(f"Unique file contents: {len(hash_to_files)}")
print(f"Duplicate groups: {sum(1 for files in hash_to_files.values() if len(files) > 1)}")
print(f"Files belonging to duplicate groups: {len(duplicate_groups)}")
print(f"Report saved to: {duplicates_path}")

if missing_files:
    print("\nMissing file examples:")
    for path in missing_files[:10]:
        print(path)

print("\nScan complete. Original images were not modified.")
