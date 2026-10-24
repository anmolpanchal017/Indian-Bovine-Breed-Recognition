
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from configs.config import (
    TEST_CSV,
    CHECKPOINT_DIR,
    REPORT_DIR,
    BATCH_SIZE,
)
from src.data.dataset import BovineDataset, EVAL_TRANSFORM
from src.training.model import create_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / "best_model_finetuned.pth"

    title="Fine-Tuned Model — Test Set Confusion Matrix",

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: breed for breed, idx in class_to_idx.items()}

    test_data = BovineDataset(
        TEST_CSV,
        transform=EVAL_TRANSFORM,
        class_to_idx=class_to_idx,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test images: {len(test_data)}")
    print(f"Test accuracy: {accuracy:.2%}")

    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_path = REPORT_DIR / "test_classification_report.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Save confusion matrix as CSV.
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(REPORT_DIR / "test_confusion_matrix.csv")

    # Save a visual confusion matrix.
    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        title="Test Set Confusion Matrix",
        xlabel="Predicted breed",
        ylabel="True breed",
    )
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(target_names, rotation=90, fontsize=7)
    ax.set_yticklabels(target_names, fontsize=7)

    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "test_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "test_images": len(test_data),
        "test_accuracy": accuracy,
        "num_classes": len(class_to_idx),
        "checkpoint": str(checkpoint_path),
    }

    with open(REPORT_DIR / "test_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("Classification report:", report_path)
    print("Confusion matrix:", REPORT_DIR / "test_confusion_matrix.png")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
