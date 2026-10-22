"""
ELEC 475: Lab 2 - Modernized LeNet Architecture
Ablation Study 1 & 2: ReLU + Overlapping Max Pooling.
"""

import torch
import torch.nn as nn


class LeNetModern(nn.Module):
    """
    Modernized LeNet architecture using:
        - ReLU activations
        - 3x3 overlapping Max Pooling with stride 2
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNetModern, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement the Modernized Feature Extractor
        #
        # Requirements:
        #   1. Conv1: in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2
        #   2. Activation: nn.ReLU(inplace=True)  [Ablation 1]
        #   3. Overlapping MaxPool1: kernel_size=3, stride=2  [Ablation 2]
        #   4. Conv2: in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2
        #   5. Activation: nn.ReLU(inplace=True)  [Ablation 1]
        #   6. Overlapping MaxPool2: kernel_size=3, stride=2  [Ablation 2]
        # ======================================================================

        self.feature_extractor = nn.Sequential(
            # ======================================================================
            # TODO [IN-LAB STEP 4]: Implement the Modernized Classifier Head
            #
            # Requirements:
            #   1. FC1: in_features = 16 * 15 * 15 (3,600), out_features = 120
            #   2. Activation: nn.ReLU(inplace=True)
            #   3. FC2: in_features = 120, out_features = 84
            #   4. Activation: nn.ReLU(inplace=True)
            #   5. FC3 (Output): in_features = 84, out_features = num_classes
            # ======================================================================

            nn.Conv2d(in_channels=in_channels,out_channels=6,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement the Modernized Classifier Head
        #
        # Requirements:
        #   1. FC1: in_features = 16 * 15 * 15 (3,600), out_features = 120
        #   2. Activation: nn.ReLU(inplace=True)
        #   3. FC2: in_features = 120, out_features = 84
        #   4. Activation: nn.ReLU(inplace=True)
        #   5. FC3 (Output): in_features = 84, out_features = num_classes
        # ======================================================================

        self.classifier = nn.Sequential(
            # Replace with dense layers using ReLU:
            # nn.Linear(...),
            # nn.ReLU(inplace=True),
            # ...

            nn.Linear(in_features=16 * 15 * 15,out_features=120),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=120,out_features=84),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=84,out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [IN-LAB STEP 4]: Implement Forward Pass
        # ======================================================================

        x = self.feature_extractor(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = LeNetModern(num_classes=10, in_channels=3)
    dummy_input = torch.randn(1, 3, 64, 64)

    print("Testing Modernized LeNet Architecture:")

    try:
        summary(model, input_size=(1, 3, 64, 64))
    except Exception as e:
        print(f"Model not fully implemented yet: {e}")
