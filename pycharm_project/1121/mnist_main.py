import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from model import MNISTClassifier
from utils import get_device, train_MNIST, vis_losses_accs, get_mnist_dataset

# MNIST data 생성
BATCH_SIZE = 32
dataloader, N_SAMPLES = get_mnist_dataset(BATCH_SIZE=BATCH_SIZE)

# hyper parameter
DEVICE = get_device()
EPOCHS = 10
LR = 0.1

# 모델 인스턴스 생성
model = MNISTClassifier().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)

losses, accs = [], []
# 학습
for epoch in tqdm(range(EPOCHS), "Train in progress", mininterval=0.1, maxinterval=1):
    epoch_loss, epoch_acc = train_MNIST(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE)

    losses.append(epoch_loss)
    accs.append(epoch_acc)

    print(f"Epoch: {epoch + 1}")
    print(f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}\n")

vis_losses_accs(losses, accs)
