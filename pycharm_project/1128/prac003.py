# using conv block make vgg 11, 13, 19
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ConvBlock, self).__init__()

        self.layers = []

        for _ in range(n_layers):
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))

            self.layers.append(nn.ReLU())
            in_channels = out_channels

        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        self.layer1 = ConvBlock(in_channels=3, out_channels=64, n_layers=2)
        self.layer2 = ConvBlock(in_channels=64, out_channels=128, n_layers=2)
        self.layer3 = ConvBlock(in_channels=128, out_channels=256, n_layers=2)
        self.layer4 = ConvBlock(in_channels=256, out_channels=512, n_layers=2)
        self.layer5 = ConvBlock(in_channels=512, out_channels=512, n_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)
        x = self.layer5.forward(x)

        x = self.classifier(x)

        return x
