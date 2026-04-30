from pathlib import Path
import torch
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CLASSES = ["cardboard", "plastic", "metal"]
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"