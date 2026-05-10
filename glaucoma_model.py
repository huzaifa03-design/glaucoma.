import torch
import torch.nn as nn
from torchvision import models


def get_glaucoma_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Return a ResNet18 model adapted for binary glaucoma classification."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    device = get_device()
    model = get_glaucoma_resnet18().to(device)
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
