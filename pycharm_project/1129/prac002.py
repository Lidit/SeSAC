# GoogLeNet (Dimension Reductions)
import torch.nn as nn
import torch
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)

        return x


class InceptionDR(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionDR, self).__init__()

        self.branch1 = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=ch1x1)

        self.branch2 = nn.Sequential(
            ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=ch3x3red),
            ConvBlock(kernel_size=3, in_channels=ch3x3red, out_channels=ch3x3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=ch5x5red),
            ConvBlock(kernel_size=5, in_channels=ch5x5red, out_channels=ch5x5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=pool_proj)
        )

    def forward(self, x):
        x1 = self.branch1.forward(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.concat([x1, x2, x3, x4], dim=1)

        return out


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(kernel_size=7, in_channels=3, out_channels=64, padding=3, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(kernel_size=1, in_channels=64, out_channels=192, padding=0, stride=1)
        self.conv3 = ConvBlock(kernel_size=3, in_channels=192, out_channels=192, padding=1, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionDR(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
        self.inception3b = InceptionDR(in_channels=256, ch1x1=128, ch3x3red=128, ch3x3=192, ch5x5red=32, ch5x5=96, pool_proj=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionDR(in_channels=480, ch1x1=192, ch3x3red=96, ch3x3=208, ch5x5red=16, ch5x5=48, pool_proj=64)

        self.sub_fc1_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        )

        self.sub_fc1_2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=1000)
        )
        self.inception4b = InceptionDR(in_channels=512, ch1x1=160, ch3x3red=112, ch3x3=224, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4c = InceptionDR(in_channels=512, ch1x1=128, ch3x3red=128, ch3x3=256, ch5x5red=24, ch5x5=64, pool_proj=64)
        self.inception4d = InceptionDR(in_channels=512, ch1x1=112, ch3x3red=144, ch3x3=288, ch5x5red=32, ch5x5=64, pool_proj=64)

        self.sub_fc2_1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(in_channels=528, out_channels=128, kernel_size=1, stride=1)
        )
        self.sub_fc2_2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=1000)
        )

        self.inception4e = InceptionDR(in_channels=528, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionDR(in_channels=832, ch1x1=256, ch3x3red=160, ch3x3=320, ch5x5red=32, ch5x5=128, pool_proj=128)
        self.inception5b = InceptionDR(in_channels=832, ch1x1=384, ch3x3red=192, ch3x3=384, ch5x5red=48, ch5x5=128, pool_proj=128)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.fc = nn.Linear(in_features=1024, out_features=1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)

        x = self.inception4a(x)
        # fc1_out = self.sub_fc1_1(x)
        # fc1_out = self.sub_fc1_2(fc1_out.view(fc1_out.size(0), -1))
        # print("fc1_out shape: ",fc1_out.shape)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        # fc2_out = self.sub_fc2_1(x)
        # fc2_out = self.sub_fc2_2(fc2_out.view(fc2_out.size(0), -1))
        # print("fc2_out shape: ", fc2_out.shape)
        x = self.inception4e(x)
        x = self.max_pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)

        x = self.fc(x.view(x.size(0), -1))

        return x


BATCH_SIZE = 32
C = 3
H, W = 224, 224

model = GoogLeNet()

print(summary(model, input_size=(C, H, W)))
