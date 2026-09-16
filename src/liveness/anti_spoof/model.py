import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class AntiSpoofCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = mobilenet_v3_small(
            weights=None
        )

        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            2
        )

    def forward(self, x):
        return self.model(x)