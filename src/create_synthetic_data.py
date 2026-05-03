import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SOURCE_DIR = Path("data/train")
TARGET_DIR = Path("data/synthetic_train")

CLASSES = ["cardboard", "plastic", "metal", "glass"]
NUM_SYNTHETIC_PER_CLASS = 70

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
RANDOM_SEED = 42


def get_images(class_dir):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(class_dir.glob(f"*{ext}"))
        images.extend(class_dir.glob(f"*{ext.upper()}"))
    return images


def apply_synthetic_transform(img):
    img = img.convert("RGB")

    choice = random.choice(["blur", "brightness", "occlusion", "rotate", "contrast"])

    if choice == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))

    elif choice == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.5, 1.5))

    elif choice == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.6, 1.6))

    elif choice == "rotate":
        img = img.rotate(random.uniform(-20, 20))

    elif choice == "occlusion":
        w, h = img.size
        box_w = int(w * random.uniform(0.15, 0.35))
        box_h = int(h * random.uniform(0.15, 0.35))
        x = random.randint(0, max(1, w - box_w))
        y = random.randint(0, max(1, h - box_h))

        patch = Image.new("RGB", (box_w, box_h), color=(0, 0, 0))
        img.paste(patch, (x, y))

    return ImageOps.exif_transpose(img)


def main():
    random.seed(RANDOM_SEED)

    if TARGET_DIR.exists():
        import shutil
        shutil.rmtree(TARGET_DIR)

    for cls in CLASSES:
        source_class_dir = SOURCE_DIR / cls
        target_class_dir = TARGET_DIR / cls
        target_class_dir.mkdir(parents=True, exist_ok=True)

        images = get_images(source_class_dir)

        if len(images) == 0:
            raise ValueError(f"No images found in {source_class_dir}")

        for i in range(NUM_SYNTHETIC_PER_CLASS):
            img_path = random.choice(images)
            img = Image.open(img_path)

            synthetic_img = apply_synthetic_transform(img)
            synthetic_img.save(target_class_dir / f"{cls}_synthetic_{i:03d}.jpg")

        print(f"Created {NUM_SYNTHETIC_PER_CLASS} synthetic images for {cls}")

    print("Synthetic data creation complete.")


if __name__ == "__main__":
    main()