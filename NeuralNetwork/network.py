import torch
from torch import nn
from torchvision.models import resnet152, ResNet152_Weights


class AnimalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        x = self.model(x)
        return x