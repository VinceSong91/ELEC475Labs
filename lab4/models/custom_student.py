"""Custom lightweight segmentation student: MobileNetV3 encoder + gated depthwise context + dual-skip decoder."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def block(ci, co, k=3):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=k//2, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class DepthwiseContext(nn.Module):
    def __init__(self, in_channels=576, width=64):
        super().__init__()
        self.reduce = block(in_channels, width, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d,
                                    groups=width, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            for d in (1, 3, 6)])
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(width, 3, 1), nn.Softmax(dim=1))
        self.project = block(width, width, 1)

    def forward(self, x):
        x = self.reduce(x)
        weights = self.gate(x)
        fused = sum(weights[:, i:i+1] * branch(x)
                    for i, branch in enumerate(self.branches))
        return self.project(fused + x)


class CustomStudentNet(nn.Module):
    """Outputs the same dictionary keys as StudentNet for training and KD."""
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        self.low = features[:4]      # 24 channels, stride 8
        self.mid = features[4:7]     # 40 channels, stride 16
        self.high = features[7:]     # 576 channels, stride 32
        self.context = DepthwiseContext(576, 64)
        self.mid_proj = block(40, 24, 1)
        self.low_proj = block(24, 16, 1)
        self.mid_fuse = block(64 + 24, 64)
        self.low_fuse = block(64 + 16, 48)
        self.dropout = nn.Dropout2d(0.2)
        self.classifier = nn.Conv2d(48, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.context(high)
        y = F.interpolate(y, size=mid.shape[-2:], mode='bilinear', align_corners=False)
        y = self.mid_fuse(torch.cat((y, self.mid_proj(mid)), dim=1))
        y = F.interpolate(y, size=low.shape[-2:], mode='bilinear', align_corners=False)
        y = self.low_fuse(torch.cat((y, self.low_proj(low)), dim=1))
        y = self.classifier(self.dropout(y))
        return {'out': F.interpolate(y, size=size, mode='bilinear', align_corners=False),
                'low': low, 'mid': mid, 'high': high}


if __name__ == '__main__':
    model = CustomStudentNet(pretrained=False).eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 320, 320))
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    for key, value in result.items():
        print(key, tuple(value.shape))
