
from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")

# 1. Find breeds with the lowest F1-score
report = pd.read_csv(
    REPORT_DIR / "test_classification_report.csv",
    index_col=0,
)

class_report = report.loc[
    ~report.index.isin(["accuracy", "macro avg", "weighted avg"])
].copy()

class_report = class_report.sort_values("f1-score")

print("\n=== 10 BREEDS NEEDING MOST IMPROVEMENT ===")
print(
    class_report[["precision", "recall", "f1-score", "support"]]
    .head(10)
    .round(3)
    .to_string()
)

# 2. Find the most common incorrect predictions
cm = pd.read_csv(
    REPORT_DIR / "test_confusion_matrix.csv",
    index_col=0,
)

pairs = []

for actual_breed in cm.index:
    for predicted_breed in cm.columns:
        if actual_breed != predicted_breed:
            count = int(cm.loc[actual_breed, predicted_breed])
            if count > 0:
                pairs.append(
                    (actual_breed, predicted_breed, count)
                )

pairs.sort(key=lambda item: item[2], reverse=True)

print("\n=== TOP 10 BREED CONFUSIONS ===")
for actual, predicted, count in pairs[:10]:
    print(
        f"Actual: {actual} | Predicted: {predicted} | "
        f"Incorrect images: {count}"
    )
