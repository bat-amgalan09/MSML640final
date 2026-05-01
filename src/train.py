import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .config import EPOCHS, LEARNING_RATE, RESULTS_DIR, DEVICE
from .dataset import get_dataloader
from .model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_curves(history, config_name):
    out_dir = RESULTS_DIR / "loss_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {config_name}")
    plt.legend()
    plt.savefig(out_dir / f"{config_name}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {config_name}")
    plt.legend()
    plt.savefig(out_dir / f"{config_name}_accuracy.png")
    plt.close()


def main(config_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    use_augmentation = config_name in ["augmentation", "synthetic_augmented"]

    use_synthetic = config_name in ["synthetic", "synthetic_augmented"]

    train_loader = get_dataloader(
        "train",
        augment=use_augmentation,
        use_synthetic=use_synthetic
    )
    val_loader = get_dataloader("val", augment=False)

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
           model, train_loader, criterion, optimizer, DEVICE
        )
 
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                RESULTS_DIR / "models" / f"{config_name}_best.pth"
            )

    plot_curves(history, config_name)

    with open(RESULTS_DIR / f"{config_name}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training complete for config: {config_name}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["baseline", "augmentation", "synthetic", "synthetic_augmented"]
    )

    args = parser.parse_args()
    main(args.config)