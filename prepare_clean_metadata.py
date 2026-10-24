
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

METADATA_PATH = ROOT / "bovine_breeds_metadata.csv"
DUPLICATES_PATH = ROOT / "reports" / "exact_duplicates.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(METADATA_PATH)
duplicates = pd.read_csv(DUPLICATES_PATH)

# Find hashes that appear under multiple breed labels.
breed_counts = duplicates.groupby("sha256")["breed"].nunique()
conflicting_hashes = set(
    breed_counts[breed_counts > 1].index
)

# Collect every file belonging to a conflicting duplicate group.
conflicting_files = duplicates[
    duplicates["sha256"].isin(conflicting_hashes)
][["breed", "path"]].drop_duplicates()

# Normalize paths to avoid Windows slash differences.
def normalize_path(path):
    return str(path).replace("\\", "/").strip().lower()

conflicting_keys = set(
    zip(
        conflicting_files["breed"].astype(str),
        conflicting_files["path"].map(normalize_path),
    )
)

metadata_keys = list(
    zip(
        metadata["breed"].astype(str),
        metadata["path"].map(normalize_path),
    )
)

mask = pd.Series(
    [key in conflicting_keys for key in metadata_keys],
    index=metadata.index,
)

clean_metadata = metadata.loc[~mask].copy()
excluded_metadata = metadata.loc[mask].copy()

clean_path = OUTPUT_DIR / "metadata_without_conflicting_duplicates.csv"
excluded_path = REPORT_DIR / "excluded_conflicting_duplicates.csv"

clean_metadata.to_csv(clean_path, index=False)
excluded_metadata.to_csv(excluded_path, index=False)

print("=" * 60)
print("CLEAN METADATA PREPARATION")
print("=" * 60)
print(f"Original metadata rows: {len(metadata)}")
print(f"Conflicting-label rows excluded: {len(excluded_metadata)}")
print(f"Remaining metadata rows: {len(clean_metadata)}")
print(f"Remaining breed classes: {clean_metadata['breed'].nunique()}")
print(f"Clean metadata saved: {clean_path}")
print(f"Excluded rows report saved: {excluded_path}")
print("\nOriginal images and original metadata were not modified.")
