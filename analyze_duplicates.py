
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DUPLICATES_PATH = PROJECT_ROOT / "reports" / "exact_duplicates.csv"

df = pd.read_csv(DUPLICATES_PATH)

if df.empty:
    print("No exact duplicate groups found.")
    raise SystemExit

# Summarize each duplicate group
group_summary = (
    df.groupby("sha256")
    .agg(
        number_of_files=("path", "size"),
        number_of_breeds=("breed", "nunique"),
        breeds=("breed", lambda x: ", ".join(sorted(set(x)))),
        paths=("path", lambda x: " | ".join(x)),
    )
    .reset_index()
)

group_summary["cross_breed"] = (
    group_summary["number_of_breeds"] > 1
)

same_breed = group_summary[~group_summary["cross_breed"]]
cross_breed = group_summary[group_summary["cross_breed"]]

REPORT_DIR = PROJECT_ROOT / "reports"

group_summary.to_csv(
    REPORT_DIR / "duplicate_group_summary.csv",
    index=False,
)

cross_breed.to_csv(
    REPORT_DIR / "cross_breed_duplicates.csv",
    index=False,
)

print("=" * 55)
print("DUPLICATE LABEL ANALYSIS")
print("=" * 55)
print(f"Total duplicate groups: {len(group_summary)}")
print(f"Same-breed duplicate groups: {len(same_breed)}")
print(f"Cross-breed duplicate groups: {len(cross_breed)}")
print(
    "Files in cross-breed groups:",
    int(cross_breed["number_of_files"].sum()),
)

if not cross_breed.empty:
    print("\nCross-breed duplicate examples:")
    print(
        cross_breed[
            ["number_of_files", "breeds", "paths"]
        ].head(15).to_string(index=False)
    )
else:
    print("\nNo cross-breed exact duplicates detected.")

print("\nReports saved in:", REPORT_DIR)
