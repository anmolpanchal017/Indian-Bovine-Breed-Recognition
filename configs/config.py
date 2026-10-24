
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT / "Indian_bovine_breeds" / "Indian_bovine_breeds"
)
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Reliable default on Windows
SEED = 42

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

for directory in [MODEL_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
