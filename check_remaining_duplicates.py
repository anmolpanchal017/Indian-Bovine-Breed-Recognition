
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parent
METADATA_PATH = (
    ROOT / "data" / "processed"
    / "metadata_without_conflicting_duplicates.csv"
)
DATASET_DIR = ROOT / "Indian_bovine_breeds" / "Indian_bovine_breeds"

df = pd.read_csv(METADATA_PATH)
hash_groups = defaultdict(list)

for row in df.itertuples(index=False):
    path = DATASET_DIR / str(row.path)

    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    hash_groups[hasher.hexdigest()].append({
        "breed": row.breed,
        "path": str(row.path),
    })

duplicate_groups = [
    items for items in hash_groups.values()
    if len(items) > 1
]

cross_breed_groups = [
    items for items in duplicate_groups
    if len({item["breed"] for item in items}) > 1
]

print("=" * 55)
print("REMAINING DATASET DUPLICATE CHECK")
print("=" * 55)
print(f"Metadata rows: {len(df)}")
print(f"Unique file contents: {len(hash_groups)}")
print(f"Same-breed duplicate groups: {len(duplicate_groups) - len(cross_breed_groups)}")
print(f"Cross-breed duplicate groups: {len(cross_breed_groups)}")
print(f"Breed classes: {df['breed'].nunique()}")
