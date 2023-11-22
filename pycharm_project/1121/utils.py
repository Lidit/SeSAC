import torch
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    return DEVICE


def train_MNIST(data, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss = 0.
    epoch_corrects = 0
    for X_, y_ in data:
        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
        X_ = X_.reshape(data.batch_size, -1)

        pred = model.forward(X_)
        loss = loss_function(pred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_)

        epoch_corrects += (torch.max(pred, axis=1)[1] == y_).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accuracy = epoch_corrects / N_SAMPLES
    return epoch_loss, epoch_accuracy


def train_classification(data, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss = 0.
    epoch_corrects = 0
    for X_, y_ in data:
        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)

        pred = model.forward(X_)
        loss = loss_function(pred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_)

        epoch_corrects += (torch.max(pred, axis=1)[1] == y_).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accuracy = epoch_corrects / N_SAMPLES
    return epoch_loss, epoch_accuracy


def get_classification_dataset(N_SAMPLES, BATCH_SIZE):
    X, y = make_blobs(n_samples=N_SAMPLES, n_features=2, cluster_std=0.7, centers=4)
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    return dataloader


def get_mnist_dataset(BATCH_SIZE):
    dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    n_samples = len(dataset)
    return dataloader, n_samples


def vis_decision_boundary(dataloader, model, DEVICE):
    ax = plt.subplot()
    x = dataloader.dataset[:][0].to(DEVICE)
    pred = model.forward(x)
    pred = torch.max(pred, axis=1)[1]
    pred = pred.cpu()
    pred = pred.detach().numpy()
    x = x.cpu()
    x = x.detach().numpy()

    ax.scatter(x[:, 0], x[:, 1], c=pred, cmap='brg')

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x1 = np.linspace(xlim[0], xlim[1], 100, dtype=np.float32)
    x2 = np.linspace(ylim[0], ylim[1], 100, dtype=np.float32)
    X1, X2 = np.meshgrid(x1, x2)

    X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    X_db = torch.tensor(X_db).to(DEVICE)

    pred_db = model.forward(X_db)
    pred_db = torch.max(pred_db, axis=1)[1]
    pred_db = pred_db.cpu()
    pred_db = pred_db.detach().numpy()

    X_db = X_db.cpu()
    X_db = X_db.detach().numpy()

    ax.scatter(X_db[:, 0], X_db[:, 1], c=pred_db, cmap='brg', alpha=0.05, zorder=-1)

    plt.show()


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
