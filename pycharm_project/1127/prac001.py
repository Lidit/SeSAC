import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.optim import SGD
import matplotlib.pyplot as plt
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv1_act = nn.Tanh()

        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv2_act = nn.Tanh()

        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.conv3_act = nn.Tanh()

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc1_act = nn.Tanh()

        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # print(x.shape)
        # b, c, h, w = x.shape
        x = self.conv1(x)
        # x = self.conv1(x.reshape(b, c, h, w))
        x = self.conv1_act(x)

        x = self.avg_pool1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)
        x = self.conv2_act(x)

        x = self.avg_pool2(x)
        print(x.shape)

        x = self.conv3(x)
        print(x.shape)
        x = self.conv3_act(x)

        x = self.fc1(x.view(x.size(0), -1))
        # print(x.shape)
        x = self.fc1_act(x)

        x = self.fc2(x)

        # print(x.shape)
        return x


class LeNetTanh(nn.Module):
    def __init__(self):
        super(LeNetTanh, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )

        if torch.cuda.is_available():
            DEVICE = 'cuda'
        elif torch.backends.mps.is_available():
            DEVICE = 'mps'
        else:
            DEVICE = 'cpu'
        self.DEVICE = DEVICE

    def forward(self, x):
        # b, c, h, w = x.shape
        x = self.layer1(x.to(DEVICE))
        # x = self.layer1(x.reshape(b, c, h, w))
        x = self.layer2(x.view(x.size(0), -1))

        return x

class LeNetRelu(nn.Module):
    def __init__(self):
        super(LeNetRelu, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

        if torch.cuda.is_available():
            DEVICE = 'cuda'
        elif torch.backends.mps.is_available():
            DEVICE = 'mps'
        else:
            DEVICE = 'cpu'

        self.DEVICE = DEVICE

    def forward(self, x):
        # b, c, h, w = x.shape
        x = self.layer1(x.to(DEVICE))
        # x = self.layer1(x.reshape(b, c, h, w))
        x = self.layer2(x.view(x.size(0), -1))

        return x


class LeNetSigmoid(nn.Module):
    def __init__(self):
        super(LeNetSigmoid, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

        if torch.cuda.is_available():
            DEVICE = 'cuda'
        elif torch.backends.mps.is_available():
            DEVICE = 'mps'
        else:
            DEVICE = 'cpu'

        self.DEVICE = DEVICE

    def forward(self, x):
        # b, c, h, w = x.shape
        x = self.layer1(x.to(DEVICE))
        # x = self.layer1(x.reshape(b, c, h, w))
        x = self.layer2(x.view(x.size(0), -1))

        return x


def get_mnist_dataset(BATCH_SIZE):
    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    n_samples = len(dataset)
    return dataloader, n_samples


def train_MNIST(data, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss = 0.
    epoch_corrects = 0
    for X_, y_ in data:
        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
        # X_ = X_.reshape(data.batch_size, -1)

        # y_ = y_.reshape(data.batch_size, -1)
        # print("y_: " ,y_.shape)

        pred = model.forward(X_)
        # print("pred: ",pred.shape)
        loss = loss_function(pred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_)

        epoch_corrects += (torch.max(pred, axis=1)[1] == y_).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accuracy = epoch_corrects / N_SAMPLES
    return epoch_loss, epoch_accuracy


def vis_losses_accs(losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)

    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)

    fig.tight_layout()
    plt.show()


# H, W = 28, 28
# input_tensor = torch.rand(size=(32, H, W))
# print(input_tensor.size(0))
# print(input_tensor.size(1))
# print(input_tensor.size(2))
#
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
torch.manual_seed(0)
print("running on ", DEVICE)
model = LeNet()
# model.forward(input_tensor)
# MNIST data 생성
EPOCHS = 1
LR = 0.1
BATCH_SIZE = 32
dataloader, N_SAMPLES = get_mnist_dataset(BATCH_SIZE=BATCH_SIZE)

# 모델 인스턴스 선언
# DEVICE = 'cpu'
model_tanh = LeNetTanh().to(DEVICE)
model_relu = LeNetRelu().to(DEVICE)
model_sigmoid = LeNetSigmoid().to(DEVICE)
loss_function_tanh = nn.CrossEntropyLoss()
loss_function_relu = nn.CrossEntropyLoss()
loss_function_sigmoid = nn.CrossEntropyLoss()
optimizer_tanh = SGD(model_tanh.parameters(), lr=LR)
optimizer_relu = SGD(model_relu.parameters(), lr=LR)
optimizer_sigmoid = SGD(model_sigmoid.parameters(), lr=LR)
# torch summary
# b,c,h,w = dataloader.dataset[0].shape
# print(dataloader.dataset[:][0].shape)
print(summary(model, input_size=(1, 28, 28), batch_size=32))
print(summary(model_tanh, input_size=(1, 28, 28), batch_size=32))
print(summary(model_relu, input_size=(1, 28, 28), batch_size=32))
print(summary(model_sigmoid, input_size=(1, 28, 28), batch_size=32))

losses_tanh, accs_tanh = [], []
losses_relu, accs_relu = [], []
losses_sigmoid, accs_sigmoid = [], []

for epoch in tqdm(range(EPOCHS), "Train in progress", mininterval=0.1, maxinterval=1):
    epoch_loss_tanh, epoch_acc_tanh = train_MNIST(dataloader, N_SAMPLES, model_tanh, loss_function_tanh, optimizer_tanh, DEVICE)
    epoch_loss_relu, epoch_acc_relu = train_MNIST(dataloader, N_SAMPLES, model_relu, loss_function_relu, optimizer_relu, DEVICE)
    epoch_loss_sigmoid, epoch_acc_sigmoid = train_MNIST(dataloader, N_SAMPLES, model_sigmoid, loss_function_sigmoid, optimizer_sigmoid, DEVICE)

    losses_tanh.append(epoch_loss_tanh)
    accs_tanh.append(epoch_acc_tanh)
    losses_relu.append(epoch_loss_relu)
    accs_relu.append(epoch_acc_relu)
    losses_sigmoid.append(epoch_loss_sigmoid)
    accs_sigmoid.append(epoch_acc_sigmoid)

    print(f"Epoch: {epoch + 1}")
    print(f"Tanh Loss: {epoch_loss_tanh:.4f} - Accuracy: {epoch_acc_tanh:.4f}\n")
    print(f"ReLU Loss: {epoch_loss_relu:.4f} - Accuracy: {epoch_acc_relu:.4f}\n")
    print(f"sigmoid Loss: {epoch_loss_sigmoid:.4f} - Accuracy: {epoch_acc_sigmoid:.4f}\n")

vis_losses_accs(losses_tanh, accs_tanh)
vis_losses_accs(losses_relu, accs_relu)
vis_losses_accs(losses_sigmoid, accs_sigmoid)
