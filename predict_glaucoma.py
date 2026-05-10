import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from glaucoma_model import get_glaucoma_resnet18, get_device


def load_checkpoint(model_path: str, device: torch.device):
    """Load model state and label mapping from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location=device)
    model = get_glaucoma_resnet18(num_classes=2, pretrained=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        class_to_idx = checkpoint.get("idx_to_class") or checkpoint.get("class_to_idx")
        if class_to_idx is None:
            class_to_idx = {"glaucoma": 0, "normal": 1}
        if "class_to_idx" in checkpoint and "idx_to_class" not in checkpoint:
            class_to_idx = checkpoint["class_to_idx"]
    else:
        model.load_state_dict(checkpoint)
        class_to_idx = {"glaucoma": 0, "normal": 1}

    if all(isinstance(k, int) for k in class_to_idx.keys()):
        idx_to_class = {k: v for k, v in class_to_idx.items()}
    else:
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply preprocessing to the image used during training."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def predict_image(model: torch.nn.Module, image: Image.Image, idx_to_class: dict):
    """Predict the image class and return label, confidence, and mapping."""
    device = get_device()
    model = model.to(device)
    model.eval()

    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, dim=0)

    label = idx_to_class[predicted.item()]
    return {
        "label": label,
        "confidence": float(confidence.item() * 100),
        "probabilities": probabilities.cpu().tolist(),
        "idx_to_class": idx_to_class,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict glaucoma from a retinal image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="glaucoma_model.pth",
        help="Path to the trained model checkpoint.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = get_device()
    model, idx_to_class = load_checkpoint(str(model_path), device)
    print(f"Detected class-to-index mapping: {idx_to_class}")

    image = Image.open(args.image_path).convert("RGB")
    result = predict_image(model, image, idx_to_class)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Probabilities: {result['probabilities']}")
