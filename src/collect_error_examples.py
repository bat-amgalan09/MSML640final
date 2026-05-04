import argparse
import csv
import shutil
import torch
from PIL import Image, ImageDraw
from torchvision import datasets

from .config import DEVICE, DATA_DIR, RESULTS_DIR
from .dataset import get_transforms
from .model import build_model


def add_label_banner(img, true_label, pred_label, confidence):
    """
    Adds a simple text banner to the image so the saved error example is easy to interpret.
    """
    img = img.convert("RGB")
    w, h = img.size

    banner_h = 45
    new_img = Image.new("RGB", (w, h + banner_h), color=(255, 255, 255))
    new_img.paste(img, (0, banner_h))

    draw = ImageDraw.Draw(new_img)
    text = f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2f}"
    draw.text((10, 12), text, fill=(0, 0, 0))

    return new_img


def collect_error_examples(config_name="baseline", max_examples=20):
    out_dir = RESULTS_DIR / "error_examples" / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model().to(DEVICE)
    model_path = RESULTS_DIR / "models" / f"{config_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "test",
        transform=get_transforms(augment=False)
    )

    class_names = test_dataset.classes

    rows = []
    error_count = 0

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image_tensor, true_idx = test_dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

            pred_idx = pred_idx.item()
            confidence = confidence.item()

            if pred_idx != true_idx:
                image_path, _ = test_dataset.samples[idx]

                true_label = class_names[true_idx]
                pred_label = class_names[pred_idx]

                original_img = Image.open(image_path).convert("RGB")
                labeled_img = add_label_banner(
                    original_img,
                    true_label,
                    pred_label,
                    confidence
                )

                save_name = (
                    f"{error_count + 1:03d}_"
                    f"true_{true_label}_pred_{pred_label}_"
                    f"conf_{confidence:.2f}.jpg"
                )

                save_path = out_dir / save_name
                labeled_img.save(save_path)

                rows.append({
                    "example_id": error_count + 1,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": round(confidence, 4),
                    "source_image": str(image_path),
                    "saved_image": str(save_path)
                })

                error_count += 1

                if error_count >= max_examples:
                    break

    csv_path = out_dir / "misclassified_examples.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "true_label",
                "predicted_label",
                "confidence",
                "source_image",
                "saved_image"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {error_count} misclassified examples to: {out_dir}")
    print(f"Saved CSV summary to: {csv_path}")

    if error_count == 0:
        print("No misclassified examples found for this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        choices=["baseline", "augmentation", "synthetic", "synthetic_augmented"]
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20
    )

    args = parser.parse_args()
    collect_error_examples(args.config, args.max_examples)