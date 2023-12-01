# ResNet34
import torch.nn as nn


# path1: X to CRC, path2: X, result: path1 + path2
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        )

        if stride != 1:
            self.path2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.path2 = nn.Identity()

        self.act = nn.ReLU()

    def forward(self, x):
        fx = self.path1(x)
        x = self.path2(x)

        result = self.act(fx + x)
        return result


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU()
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self._make_layers(in_channels=64, out_channels=64, n_blocks=3, downsample=False)
        self.conv3 = self._make_layers(in_channels=64, out_channels=128, n_blocks=4, downsample=True)
        self.conv4 = self._make_layers(in_channels=128, out_channels=256, n_blocks=6, downsample=True)
        self.conv5 = self._make_layers(in_channels=256, out_channels=512, n_blocks=3, downsample=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def _make_layers(self, in_channels, out_channels, n_blocks, downsample):
        stride = 2 if downsample else 1
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to('mps')
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = self.fc(x.view(x.size(0), -1))

        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = ResNet34().to('mps')
    summary(model, input_size=(3, 224, 224))
