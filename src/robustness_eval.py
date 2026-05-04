import argparse
import csv
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .config import DEVICE, DATA_DIR, RESULTS_DIR
from .dataset import get_transforms
from .model import build_model


class RobustnessDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, perturbation="clean"):
        self.base_dataset = datasets.ImageFolder(root=root_dir)
        self.perturbation = perturbation

        # Important: use the same preprocessing as normal evaluation.
        # This respects USE_IMAGENET_NORM=0 or USE_IMAGENET_NORM=1.
        self.transform = get_transforms(augment=False)

    def __len__(self):
        return len(self.base_dataset)

    def apply_perturbation(self, img):
        img = img.convert("RGB")

        if self.perturbation == "clean":
            return img

        if self.perturbation == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2.0))

        if self.perturbation == "dark":
            return ImageEnhance.Brightness(img).enhance(0.55)

        if self.perturbation == "bright":
            return ImageEnhance.Brightness(img).enhance(1.45)

        if self.perturbation == "occlusion":
            img = img.copy()
            w, h = img.size
            draw = ImageDraw.Draw(img)
            box_w, box_h = int(w * 0.25), int(h * 0.25)
            x = random.randint(0, max(1, w - box_w))
            y = random.randint(0, max(1, h - box_h))
            draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))
            return img

        if self.perturbation == "noise":
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 25, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        raise ValueError(f"Unknown perturbation: {self.perturbation}")

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.apply_perturbation(img)
        img = self.transform(img)
        return img, label


def evaluate_under_perturbation(model, perturbation):
    dataset = RobustnessDataset(DATA_DIR / "test", perturbation=perturbation)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds)


def main(config_name):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = build_model().to(DEVICE)
    model_path = RESULTS_DIR / "models" / f"{config_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    perturbations = ["clean", "blur", "noise", "dark", "bright", "occlusion"]

    out_dir = RESULTS_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for perturbation in perturbations:
        acc = evaluate_under_perturbation(model, perturbation)
        rows.append({
            "model": config_name,
            "perturbation": perturbation,
            "accuracy": round(acc, 4)
        })
        print(f"{perturbation}: {acc:.4f}")

    csv_path = out_dir / f"{config_name}_robustness_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "perturbation", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved robustness results to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="synthetic",
        choices=["baseline", "augmentation", "synthetic", "synthetic_augmented"]
    )

    args = parser.parse_args()
    main(args.config)