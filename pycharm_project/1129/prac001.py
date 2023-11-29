# Branch Implementation
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


class InceptionNaive(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5):
        super(InceptionNaive, self).__init__()

        self.branch1 = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=ch1x1)

        self.branch2 = ConvBlock(kernel_size=3, in_channels=in_channels, out_channels=ch3x3, padding=1)

        self.branch3 = ConvBlock(kernel_size=5, in_channels=in_channels, out_channels=ch5x5, padding=2)

        self.branch4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x1 = self.branch1.forward(x)
        x2 = self.branch2.forward(x)
        x3 = self.branch3.forward(x)
        x4 = self.branch4.forward(x)

        out = torch.concat([x1, x2, x3, x4], axis=1)

        return out


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

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


BATCH_SIZE = 32
C = 192
H, W = 100, 100
input_tensor = torch.rand((BATCH_SIZE, C, H, W))

# model1 = InceptionNaive(in_channels=192, ch1x1=64, ch3x3=128, ch5x5=32)
model2 = Inception(in_channels=192, ch1x1=64, ch3x3red=96, ch3x3=128, ch5x5red=16, ch5x5=32, pool_proj=32)
print(summary(model2, input_size=(C, H, W), batch_size=BATCH_SIZE))

output = model2.forward(input_tensor)
print("output shape of InceptionDR: ", output.shape)

# print(summary(model1, input_size=(192, 100, 100), batch_size=32))

# (192, 100, 100) image
# to (1,1) convolution (kernel_size=1, in_channels=192, out_channels=64, padding=0, stride=1) -> 64,100,100 image
# to (3,3) convolution (kernel_size=3, in_channels=192, out_channels=128, padding=1, stride=1) -> 128,100,100 image
# to (5,5) convolution (kernel_size=5, in_channels=192, out_channels=32, padding=2, stride=1) -> 32,100,100 image
# to (3,3) max pooling (kernel_size=3, padding=1,stride=1) -> 192, 100, 100 image

# input_tensor = torch.rand((192, 100, 100))
#
# branch1 = ConvBlock(kernel_size=1, in_channels=192, out_channels=64, padding=0, stride=1)
# out_branch1 = branch1.forward(input_tensor)
#
# branch2 = ConvBlock(kernel_size=3, in_channels=192, out_channels=128, padding=1, stride=1)
# out_branch2 = branch2.forward(input_tensor)
#
# branch3 = ConvBlock(kernel_size=5, in_channels=192, out_channels=32, padding=2, stride=1)
# out_branch3 = branch3.forward(input_tensor)
#
# branch4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
# out_branch4 = branch4(input_tensor)
#
# out_inception = torch.concat([out_branch1,out_branch2, out_branch3, out_branch4], axis=0)
#
# print("branch1: ", out_branch1.shape)
# print("branch2: ", out_branch2.shape)
# print("branch3: ", out_branch3.shape)
# print("branch4: ", out_branch4.shape)
# print("out_inception: ", out_inception.shape)


# print(summary(branch1, input_size=(192, 100, 100)))
# print(summary(branch2, input_size=(192, 100, 100)))
# print(summary(branch3, input_size=(192, 100, 100)))
# print(summary(branch4, input_size=(192, 100, 100)))
