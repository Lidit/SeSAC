# conv block
import torch.nn as nn
from torchsummary import summary


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


# block = ConvBlock(in_channels=3, out_channels=64, n_layers=1)
#
# summary(block, input_size=(3, 100, 100))

block = ConvBlock(in_channels=3, out_channels=64, n_layers=3)
summary(block, input_size=(3, 100, 100))
