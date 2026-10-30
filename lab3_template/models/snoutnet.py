"""
ELEC 475: Lab 3 - SnoutNet Nose Localization Models
Three regression models that predict a pet's nose position (u, v):
    SnoutNet        : Custom 3-block CNN trained from scratch
    SnoutNetAlexNet : SnoutNet-A, ImageNet-pretrained torchvision AlexNet backbone
    SnoutNetVGG16   : SnoutNet-V, ImageNet-pretrained torchvision VGG16 backbone

All models take (B, 3, 227, 227) input and return (B, 2) NORMALIZED coordinates
(u/227, v/227). Layer names must match exactly for the provided weights to load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, vgg16, AlexNet_Weights, VGG16_Weights


class SnoutNet(nn.Module):
    """
    SnoutNet Architecture for (3, 227, 227) RGB images.

    Spatial dimension arithmetic (pooling with ceil_mode=True rounds UP):
        Input        : (B, 3, 227, 227)
        Conv1 (p=1)  : (B, 64, 227, 227)
        MaxPool1     : ceil((227 - 4)/4) + 1 = 57 --> (B, 64, 57, 57)
        Conv2 (p=1)  : (B, 128, 57, 57)
        MaxPool2     : ceil((57 - 4)/4) + 1 = 15  --> (B, 128, 15, 15)
        Conv3 (p=1)  : (B, 256, 15, 15)
        MaxPool3     : ceil((15 - 4)/4) + 1 = 4   --> (B, 256, 4, 4)
        Flatten      : 256 * 4 * 4 = 4,096 features
        FC1          : 4,096 -> 1,024
        FC2          : 1,024 -> 1,024
        FC3          : 1,024 -> 2
    """
    def __init__(self):
        super(SnoutNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # One pooling layer shared by all three conv blocks
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 64, 57, 57)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 128, 15, 15)
        x = self.pool(F.relu(self.conv3(x)))   # (B, 256, 4, 4)
        x = torch.flatten(x, start_dim=1)      # (B, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                     # (B, 2) normalized (u, v)


class SnoutNetAlexNet(nn.Module):
    """
    SnoutNet-A: torchvision AlexNet with its final 1000-class layer replaced
    by a 2-output regression layer.
    """
    def __init__(self, pretrained: bool = True):
        super(SnoutNetAlexNet, self).__init__()

        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.backbone = alexnet(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SnoutNetVGG16(nn.Module):
    """
    SnoutNet-V: torchvision VGG16 with its final 1000-class layer replaced
    by a 2-output regression layer.
    """
    def __init__(self, pretrained: bool = True):
        super(SnoutNetVGG16, self).__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torchinfo import summary

    x = torch.randn(1, 3, 227, 227)

    # Layer-by-layer shapes for SnoutNet (compare with the Step 1 table)
    summary(SnoutNet(), input_size=(1, 3, 227, 227))

    # pretrained=False: same shapes and params, and it skips the ImageNet download
    for name, model in [("SnoutNet", SnoutNet()),
                        ("SnoutNetAlexNet", SnoutNetAlexNet(pretrained=False)),
                        ("SnoutNetVGG16", SnoutNetVGG16(pretrained=False))]:
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"{name} output {tuple(out.shape)} params {count_params(model):,}")

    checks = [(SnoutNet(), "snoutnet.pth"), (SnoutNetAlexNet(pretrained=False), "alexnet.pth"), (SnoutNetVGG16(pretrained=False), "vgg16.pth")]
    for model, name in checks:
        state = torch.load(f"checkpoints/{name}", map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {name}")

    from utils.dataset import PetNoseDataset

    model = checks[0][0]  # SnoutNet (the loop leaves VGG16 in `model`, so index explicitly)
    dataset = PetNoseDataset(split="test")

    model.eval()
    image, label, orig_size = dataset[0]
    with torch.no_grad():
        # Model output is normalized: (u/227, v/227)
        pred = model(image.unsqueeze(0))[0]

    print("Ground truth:", label / 227 * orig_size)
    print("Prediction:  ", pred * orig_size)   

    