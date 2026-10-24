
import pandas as pd

from configs.config import TRAIN_CSV, VAL_CSV

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_counts = train_df["breed"].value_counts()
val_counts = val_df["breed"].value_counts()

summary = pd.DataFrame({
    "train_images": train_counts,
    "validation_images": val_counts,
}).fillna(0).astype(int)

summary["train_val_ratio"] = (
    summary["train_images"] /
    summary["validation_images"].replace(0, 1)
).round(2)

print("Number of breeds:", len(summary))
print("Total training records:", len(train_df))
print("Total validation records:", len(val_df))

print("\n=== FEWEST TRAINING IMAGES ===")
print(
    summary.sort_values("train_images")
    .head(10)
    .to_string()
)

print("\n=== MOST TRAINING IMAGES ===")
print(
    summary.sort_values("train_images", ascending=False)
    .head(10)
    .to_string()
)

print("\n=== BREEDS WITH FEWEST VALIDATION IMAGES ===")
print(
    summary.sort_values("validation_images")
    .head(10)
    .to_string()
)

summary.to_csv("reports/class_balance_summary.csv")
print("\nSaved: reports/class_balance_summary.csv")
