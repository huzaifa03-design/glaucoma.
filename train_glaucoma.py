import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataset_loader import get_glaucoma_dataloaders
from glaucoma_model import get_glaucoma_resnet18, get_device


def evaluate_loader(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float, torch.Tensor]:
    """Evaluate model on a DataLoader and return loss, accuracy, and confusion matrix."""
    model.eval()
    device = get_device()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for true_label, pred_label in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[true_label, pred_label] += 1

    average_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total * 100 if total > 0 else 0.0
    return average_loss, accuracy, confusion


def build_classification_report(confusion: torch.Tensor, idx_to_class: Dict[int, str]) -> List[Dict[str, float]]:
    """Create a classification report from the confusion matrix."""
    num_classes = confusion.size(0)
    report = []

    for index in range(num_classes):
        true_positive = confusion[index, index].item()
        false_positive = confusion[:, index].sum().item() - true_positive
        false_negative = confusion[index, :].sum().item() - true_positive
        support = confusion[index, :].sum().item()

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        report.append(
            {
                "Class": idx_to_class[index],
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1_score, 4),
                "Support": int(support),
            }
        )

    return report


def train_glaucoma_model(
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "glaucoma_model.pth",
):
    device = get_device()
    train_loader, val_loader, test_loader, idx_to_class = get_glaucoma_dataloaders(
        data_dir, batch_size=batch_size
    )

    model = get_glaucoma_resnet18(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_accuracy = 0.0
    history = {"train_loss": [], "val_accuracy": []}

    print(f"Detected label mapping: {idx_to_class}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        train_loss = running_loss / total_batches if total_batches > 0 else 0.0
        val_loss, val_accuracy, _ = evaluate_loader(model, val_loader, criterion, num_classes=2)

        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)

        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "idx_to_class": idx_to_class,
                    "history": history,
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": epoch,
                },
                save_path,
            )
            print(f"Saved best model at epoch {epoch} with validation accuracy {val_accuracy:.2f}%.")

    if best_val_accuracy == 0.0:
        torch.save(
            {
                "model_state": model.state_dict(),
                "idx_to_class": idx_to_class,
                "history": history,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": epochs,
            },
            save_path,
        )
        print(f"Saved final model to {save_path}")

    test_loss, test_accuracy, test_confusion = evaluate_loader(model, test_loader, criterion, num_classes=2)
    report = build_classification_report(test_confusion, idx_to_class)

    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(test_confusion)
    print("Classification Report:")
    for row in report:
        print(row)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train glaucoma detection model.")
    parser.add_argument("data_dir", type=str, default="dataset", nargs="?", help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="glaucoma_model.pth",
        help="Path to save the best model checkpoint.",
    )
    args = parser.parse_args()

    trained_model = train_glaucoma_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
    print("Training complete.")
