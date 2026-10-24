
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    REPORT_DIR,
)
from src.data.dataset import (
    BovineDataset,
    TRAIN_TRANSFORM,
    EVAL_TRANSFORM,
)
from src.training.model import create_model


# Final fine-tuning configuration
EPOCHS = 10
BACKBONE_LR = 0.0001
CLASSIFIER_LR = 0.0005
WEIGHT_DECAY = 0.0001
PATIENCE = 3

BASELINE_PATH = CHECKPOINT_DIR / "best_model.pth"
OUTPUT_PATH = CHECKPOINT_DIR / "best_model_finetuned.pth"
HISTORY_PATH = CHECKPOINT_DIR / "training_history_finetuned.json"


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {BASELINE_PATH}"
        )

    # Load the existing baseline checkpoint.
    checkpoint = torch.load(
        BASELINE_PATH,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    train_data = BovineDataset(
        TRAIN_CSV,
        transform=TRAIN_TRANSFORM,
        class_to_idx=class_to_idx,
    )
    val_data = BovineDataset(
        VAL_CSV,
        transform=EVAL_TRANSFORM,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Restore the trained baseline instead of starting from scratch.
    model = create_model(
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze the backbone first.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune only the last three feature blocks.
    for block in list(model.features.children())[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Keep the full classifier trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    backbone_parameters = [
        p for p in model.features.parameters()
        if p.requires_grad
    ]
    classifier_parameters = [
        p for p in model.classifier.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        elapsed = time.time() - start_time

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": round(elapsed, 1),
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        # Prefer higher validation accuracy; break ties by lower loss.
        improved = (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1,
                    "model_name": "MobileNetV3-Small",
                    "fine_tuned": True,
                },
                OUTPUT_PATH,
            )
        else:
            epochs_without_improvement += 1

        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    print("\nFine-tuning completed.")
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {OUTPUT_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
