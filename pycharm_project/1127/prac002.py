# VGG11
import torch.nn as nn
import torch
from torchsummary import summary


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        #
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        # self.act1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        # self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        # self.act3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        # self.act4 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        # self.act5 = nn.ReLU()
        # self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        # self.act6 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        # self.act7 = nn.ReLU()
        # self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        # self.act8 = nn.ReLU()
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        # self.act9 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.act10 = nn.ReLU()
        # self.fc3 = nn.Linear(in_features=4096, out_features=1000)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000)
        )

    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(self.layer5.eval())
        # print(x.shape)
        # print(x.view())
        x = self.fc_layer(x.view(-1, 512 * 7 * 7))

        #
        # x = self.act1(self.conv1(x))
        # x = self.pool1(x)
        #
        # x = self.act2(self.conv2(x))
        # x = self.pool2(x)
        #
        # x = self.act3(self.conv3(x))
        # x = self.act4(self.conv4(x))
        # x = self.pool3(x)
        #
        # x = self.act5(self.conv5(x))
        # x = self.act6(self.conv6(x))
        # x = self.pool4(x)
        #
        # x = self.act7(self.conv7(x))
        # x = self.act8(self.conv8(x))
        # x = self.pool5(x)
        # print(x.shape)
        #
        # x = self.act9(self.fc1(x.view(x.size(0),-1)))
        #
        # x = self.act10(self.fc2(x))
        #
        # x = self.fc3(x)

        return x


C, H, W = 3, 224, 224
input_tensor = torch.rand(size=(C, H, W))

model = VGG11()

# model.forward(input_tensor)
print(summary(model=model, input_size=(C, H, W), batch_size=1))

out =model.forward(input_tensor)
print(out.shape)
