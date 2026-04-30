from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import IMAGE_SIZE, BATCH_SIZE


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])


def get_dataloader(split="train", augment=False):
    transform = get_transforms(augment)

    dataset = datasets.ImageFolder(
        root=f"data/{split}",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train")
    )

    return loader