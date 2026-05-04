import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .config import DEVICE, RESULTS_DIR
from .dataset import get_dataloader
from .model import build_model


def evaluate_model(config_name):
    model = build_model().to(DEVICE)

    model_path = RESULTS_DIR / "models" / f"{config_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_loader = get_dataloader("test", augment=False)

    # Pull class names directly from ImageFolder so labels are always correct
    class_names = test_loader.dataset.classes
    

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"{config_name} Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(class_names)))
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    out_dir = RESULTS_DIR / "confusion_matrices"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {config_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{config_name}_cm.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["baseline", "augmentation", "synthetic", "synthetic_augmented"]
    )

    args = parser.parse_args()
    evaluate_model(args.config)