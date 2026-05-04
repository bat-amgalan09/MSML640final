from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from .config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR


def get_transforms(augment=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

def get_dataloader(split="train", augment=False, use_synthetic=False):
    transform = get_transforms(augment)

    real_dataset = datasets.ImageFolder(
        root=DATA_DIR / split,
        transform=transform
    )

    if split == "train" and use_synthetic:
        synthetic_dataset = datasets.ImageFolder(
            root=DATA_DIR / "synthetic_train",
            transform=transform
        )

        dataset = ConcatDataset([real_dataset, synthetic_dataset])
    else:
        dataset = real_dataset

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train")
    )
    print(f"Augment: {augment}, Synthetic: {use_synthetic}")
    return loader