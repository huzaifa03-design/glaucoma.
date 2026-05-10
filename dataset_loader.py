import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Tuple


def get_glaucoma_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Return train, validation, and test DataLoaders with correct label mapping."""

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset_full = datasets.ImageFolder(root=train_dir, transform=train_transform)
    class_to_idx = train_dataset_full.class_to_idx
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    print(f"Detected class-to-index mapping: {class_to_idx}")

    num_samples = len(train_dataset_full)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    if train_size <= 0:
        raise ValueError("Not enough training images found for validation split.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(
        datasets.ImageFolder(root=train_dir, transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        datasets.ImageFolder(root=train_dir, transform=eval_transform),
        val_indices,
    )
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, idx_to_class


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent / "dataset"
    train_loader, val_loader, test_loader, idx_to_class = get_glaucoma_dataloaders(root_dir)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Detected class-to-index mapping: {idx_to_class}")
    for images, labels in train_loader:
        print("Batch image tensor shape:", images.shape)
        print("Batch label tensor shape:", labels.shape)
        break
