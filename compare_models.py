
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from configs.config import VAL_CSV, CHECKPOINT_DIR, REPORT_DIR, BATCH_SIZE
from src.data.dataset import BovineDataset, EVAL_TRANSFORM
from src.training.model import create_model


def load_checkpoint(path, device):
    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def evaluate_model(checkpoint_path, val_loader, class_names, device):
    checkpoint = load_checkpoint(checkpoint_path, device)

    model = create_model(
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return accuracy_score(y_true, y_pred), report


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    baseline_path = CHECKPOINT_DIR / "best_model.pth"
    weighted_path = CHECKPOINT_DIR / "best_model_weighted.pth"

    baseline_ckpt = load_checkpoint(baseline_path, device)
    class_to_idx = baseline_ckpt["class_to_idx"]
    class_names = [
        breed for breed, idx in sorted(
            class_to_idx.items(), key=lambda item: item[1]
        )
    ]

    val_data = BovineDataset(
        VAL_CSV,
        transform=EVAL_TRANSFORM,
        class_to_idx=class_to_idx,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    baseline_acc, baseline_report = evaluate_model(
        baseline_path, val_loader, class_names, device
    )
    weighted_acc, weighted_report = evaluate_model(
        weighted_path, val_loader, class_names, device
    )

    print(f"Baseline validation accuracy: {baseline_acc:.2%}")
    print(f"Weighted validation accuracy: {weighted_acc:.2%}")

    rows = []
    for breed in class_names:
        rows.append({
            "breed": breed,
            "baseline_precision": baseline_report[breed]["precision"],
            "baseline_recall": baseline_report[breed]["recall"],
            "baseline_f1": baseline_report[breed]["f1-score"],
            "weighted_precision": weighted_report[breed]["precision"],
            "weighted_recall": weighted_report[breed]["recall"],
            "weighted_f1": weighted_report[breed]["f1-score"],
        })

    comparison = pd.DataFrame(rows)
    comparison["recall_change"] = (
        comparison["weighted_recall"] - comparison["baseline_recall"]
    )
    comparison["f1_change"] = (
        comparison["weighted_f1"] - comparison["baseline_f1"]
    )

    output_path = REPORT_DIR / "validation_model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\n=== Biggest recall improvements ===")
    print(
        comparison.sort_values("recall_change", ascending=False)
        [["breed", "baseline_recall", "weighted_recall", "recall_change"]]
        .head(10).round(3).to_string(index=False)
    )

    print("\n=== Biggest recall declines ===")
    print(
        comparison.sort_values("recall_change")
        [["breed", "baseline_recall", "weighted_recall", "recall_change"]]
        .head(10).round(3).to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
