
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("data/raw")
TARGET_DIR = Path("data")

CLASSES = ["cardboard", "plastic", "metal", "glass"]
TRAIN_PER_CLASS = 70
VAL_PER_CLASS = 15
TEST_PER_CLASS = 15
RANDOM_SEED = 42
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

def make_dirs():
    for split in ["train", "val", "test", "synthetic_train"]:
        split_dir = TARGET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (TARGET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def get_images(class_dir):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(class_dir.glob(f"*{ext}"))
        images.extend(class_dir.glob(f"*{ext.upper()}"))
    return images



def split_and_copy():
    random.seed(RANDOM_SEED)

    for cls in CLASSES:
        class_dir = SOURCE_DIR / cls
        images = get_images(class_dir)

        if len(images) < 100:
            raise ValueError(f"{cls} only has {len(images)} images. Need at least 100.")

        random.shuffle(images)

        train_imgs = images[:TRAIN_PER_CLASS]
        val_imgs = images[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
        test_imgs = images[TRAIN_PER_CLASS + VAL_PER_CLASS:
                           TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS]

        for split, split_imgs in {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }.items():
            for img in split_imgs:
                dest = TARGET_DIR / split / cls / img.name
                shutil.copy2(img, dest)

        print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")


if __name__ == "__main__":
    make_dirs()
    split_and_copy()
    print("Data preparation complete.")