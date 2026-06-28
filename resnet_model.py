import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images):
    - 3x3 conv, stride=1, no maxpool
    - final FC -> 10 classes
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)  # no pretrained weights

        # Replace first conv + maxpool for CIFAR-10
        m.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        m.maxpool = nn.Identity()

        # Replace classifier head
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        self.model = m

    def forward(self, x):
        return self.model(x)
