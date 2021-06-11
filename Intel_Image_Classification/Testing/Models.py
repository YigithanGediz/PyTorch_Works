import torch
from torch import nn
import torchvision


class ResnetTrained(nn.Module):
    def __init__(self, n_classes=6, train_resnet=False, pretrained=True):
        super().__init__()

        self.resnet = torchvision.models.resnet101(pretrained=pretrained)

        if not train_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.in_features = self.resnet.fc.in_features

        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.in_features, out_features=512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=n_classes),
            nn.LeakyReLU(),
        )

        self.fully_connected2 = nn.Linear(self.in_features, n_classes)

        self.resnet.fc = self.fully_connected


    def forward(self,x):
        return self.resnet(x)