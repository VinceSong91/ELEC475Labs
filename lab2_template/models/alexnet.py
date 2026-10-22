"""
ELEC 475: Lab 2 - Full AlexNet Architecture (Starter Template)
Adapted for 64x64 RGB input with configurable Normalization:
    - 'none': No normalization
    - 'lrn' : Local Response Normalization (original 2012 paper)
    - 'bn'  : Modern Batch Normalization (modern standard post-2015)

Reference:
    A. Krizhevsky, I. Sutskever, and G. E. Hinton,
    "ImageNet Classification with Deep Convolutional Neural Networks,"
    NeurIPS 2012.
"""
import torch
import torch.nn as nn

class AlexNet64(nn.Module):
    """
    AlexNet architecture adapted for (3, 64, 64) RGB images.
    Features 5 deep convolutional stages, overlapping MaxPool (3x3, s=2),
    configurable normalization (LRN vs BatchNorm), and dense classifier with Dropout (p=0.5).

    Spatial dimensions through the network:
        Input   : (B, 3, 64, 64)
        Conv1   : 5x5, s=2, p=2  --> (B, 64, 32, 32)
        Pool1   : 3x3, s=2       --> (B, 64, 15, 15)
        Conv2   : 5x5, s=1, p=2  --> (B, 192, 15, 15)
        Pool2   : 3x3, s=2       --> (B, 192, 7, 7)
        Conv3   : 3x3, s=1, p=1  --> (B, 384, 7, 7)
        Conv4   : 3x3, s=1, p=1  --> (B, 384, 7, 7)
        Conv5   : 3x3, s=1, p=1  --> (B, 256, 7, 7)
        Pool3   : 3x3, s=2       --> (B, 256, 3, 3)
        Flatten : 256 * 3 * 3 = 2,304 features
        FC1     : 2,304 -> 1,024 (with Dropout 0.5 & ReLU)
        FC2     : 1,024 -> 1,024 (with Dropout 0.5 & ReLU)
        FC3     : 1,024 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3, norm_type: str = "bn"):
        super(AlexNet64, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.norm_type = norm_type.lower()

        # ======================================================================
        # Helper: Normalization Layer Factory
        # ======================================================================
        def get_norm_layer(num_features: int) -> nn.Module:
            if self.norm_type == "bn":
                return nn.BatchNorm2d(num_features)
            elif self.norm_type == "lrn":
                # Original AlexNet LRN: k=2, n=5, alpha=1e-4, beta=0.75
                return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
            elif self.norm_type == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type '{norm_type}'. Choose from 'none', 'lrn', 'bn'.")

        # ======================================================================
        # TODO [POST-LAB STEP 1 & 2]: Implement the 5-Stage AlexNet Feature Extractor
        #
        # Build self.features using nn.Sequential or a list of layers:
        #   Stage 1: Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        #            -> get_norm_layer(64)
        #            -> nn.ReLU(inplace=True)
        #            -> nn.MaxPool2d(kernel_size=3, stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        #
        #   Stage 2: Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        #            -> get_norm_layer(192)
        #            -> nn.ReLU(inplace=True)
        #            -> nn.MaxPool2d(kernel_size=3, stride=2)

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            get_norm_layer(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        #
        #   Stage 3: Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        #            -> (nn.BatchNorm2d(384) if self.norm_type == 'bn' else nn.Identity())
        #            -> nn.ReLU(inplace=True)
        #

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384) if self.norm_type == 'bn' else nn.Identity(),
            nn.ReLU(inplace=True),

        #   Stage 4: Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        #            -> (nn.BatchNorm2d(384) if self.norm_type == 'bn' else nn.Identity())
        #            -> nn.ReLU(inplace=True)
        #
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384) if self.norm_type == 'bn' else nn.Identity(),
            nn.ReLU(inplace=True),

        #   Stage 5: Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        #            -> (nn.BatchNorm2d(256) if self.norm_type == 'bn' else nn.Identity())
        #            -> nn.ReLU(inplace=True)
        #            -> nn.MaxPool2d(kernel_size=3, stride=2)
        # ======================================================================
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256) if self.norm_type == 'bn' else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # ======================================================================
        # TODO [POST-LAB STEP 1]: Implement the Dense Classifier with Dropout
        #
        # Requirements:
        #   1. nn.Dropout(p=0.5)
        #   2. nn.Linear(in_features=256 * 3 * 3, out_features=1024)
        #   3. nn.ReLU(inplace=True)
        #   4. nn.Dropout(p=0.5)
        #   5. nn.Linear(in_features=1024, out_features=1024)
        #   6. nn.ReLU(inplace=True)
        #   7. nn.Linear(in_features=1024, out_features=num_classes)
        # ======================================================================

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*3*3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 1]: Implement the Forward Pass
        # 1. Pass input x through self.features: x = self.features(x)
        # 2. Flatten feature maps: x = torch.flatten(x, start_dim=1)
        # 3. Pass flattened vector through self.classifier: logits = self.classifier(x)
        # 4. Return logits
        # ======================================================================

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)


        return logits

if __name__ == "__main__":
    from torchinfo import summary
    model = AlexNet64(num_classes=10, in_channels=3, norm_type="bn")
    print("Testing AlexNet-64 Architecture:")
    try:
        summary(model, input_size=(1, 3, 64, 64))
    except Exception as e:
        print(f"Model not fully implemented yet: {e}")
