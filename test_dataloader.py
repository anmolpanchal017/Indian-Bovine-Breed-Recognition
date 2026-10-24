
from torch.utils.data import DataLoader

from configs.config import TRAIN_CSV, VAL_CSV, BATCH_SIZE
from src.data.dataset import BovineDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM

train_data = BovineDataset(TRAIN_CSV, transform=TRAIN_TRANSFORM)
val_data = BovineDataset(
    VAL_CSV,
    transform=EVAL_TRANSFORM,
    class_to_idx=train_data.class_to_idx,
)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

images, labels = next(iter(train_loader))

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Number of classes:", len(train_data.class_to_idx))
print("Batch image shape:", tuple(images.shape))
print("Batch label shape:", tuple(labels.shape))
print("First 5 labels:", labels[:5].tolist())
print("Data loader test passed!")
