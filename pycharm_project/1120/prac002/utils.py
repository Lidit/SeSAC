import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset, DataLoader


def get_dataset(N_SAMPLES, BATCH_SIZE):
    # # 1사분면
    # X1 = np.random.uniform(low=0.5, high=1.5, size=(50, 2))
    # # 3사분면
    # X3 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 2))
    #
    # # 2사분면
    # x2 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 1))
    # y2 = np.random.uniform(low=0.5, high=1.5, size=(50, 1))
    # X2 = np.hstack((x2, y2))
    # # 4사분면
    # x4 = np.random.uniform(low=0.5, high=1.5, size=(50, 1))
    # y4 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 1))
    # X4 = np.hstack((x4, y4))
    #
    # X = np.vstack((X1, X2, X3, X4))
    #
    # y = ((X[:, 1] * X[:, 0] < 0)).astype(int)
    # data = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    # dataset = DataLoader(data, batch_size=BATCH_SIZE)

    HALF_N_SAMPLES = int(N_SAMPLES / 2)
    # 2, 3
    centers1 = [(-1, 1), (-1, -1)]
    # 4, 1
    centers2 = [(1, -1), (1, 1)]
    X1, y1 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers1, n_features=2, cluster_std=0.3)
    X2, y2 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers2, n_features=2, cluster_std=0.3)

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2]).flatten()

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    # return dataset
    return dataloader


def get_xor_dataset(N_SAMPLES):
    HALF_N_SAMPLES = int(N_SAMPLES / 2)
    centers1 = [(1, 1), (-1, 1)]
    centers2 = [(-1, -1), (1, -1)]
    X1, y1 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers1, n_features=2, cluster_std=0.3)
    X2, y2 = make_blobs(n_samples=HALF_N_SAMPLES, centers=centers2, n_features=2, cluster_std=0.3)

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2]).flatten()
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))

    return dataset


def train(data, N_SAMPLES, model, loss_function, optimizer, DEVICE):
    epoch_loss = 0.
    epoch_corrects = 0
    for X, y in data:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model.forward(X)
        loss = loss_function(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X)

        pred = (pred > 0.5).type(torch.float)
        epoch_corrects += (pred == y).sum().item()

    epoch_loss /= N_SAMPLES
    epoch_accuracy = epoch_corrects / N_SAMPLES
    return epoch_loss, epoch_accuracy


def get_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    return DEVICE


def vis_losses_accs(losses, accs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("BCE Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)

    fig.tight_layout()
    plt.show()


def vis_decision_boundary(x, model, DEVICE):
    # plot dataset
    ax = plt.subplot()
    x = x.to(DEVICE)
    pred = model.forward(x)
    pred = (pred > 0.5).type(torch.float)
    pred = pred.cpu()
    pred = pred.detach().numpy()

    x = x.cpu()
    x = x.detach().numpy()

    ax.scatter(x[:, 0], x[:, 1], c=pred, cmap='bwr')
    # plot mesh grid
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x1 = np.linspace(xlim[0], xlim[1], 100, dtype=np.float32)
    x2 = np.linspace(ylim[0], ylim[1], 100, dtype=np.float32)
    X1, X2 = np.meshgrid(x1, x2)
    X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])

    X_db = torch.tensor(X_db).to(DEVICE)
    pred_db = model.forward(X_db)
    pred_db = (pred_db > 0.5).type(torch.float)
    pred_db = pred_db.cpu()
    pred_db = pred_db.detach().numpy()

    X_db = X_db.cpu()
    X_db = X_db.detach().numpy()
    ax.scatter(X_db[:, 0], X_db[:, 1], c=pred_db, cmap='bwr', alpha=0.05, zorder=-1)
    plt.show()


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(5, 5))
    # 1사분면
    X1 = np.random.uniform(low=0.5, high=1.5, size=(50, 2))
    # 3사분면
    X3 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 2))

    # 2사분면
    x2 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 1))
    y2 = np.random.uniform(low=0.5, high=1.5, size=(50, 1))
    X2 = np.hstack((x2, y2))
    # 4사분면
    x4 = np.random.uniform(low=0.5, high=1.5, size=(50, 1))
    y4 = np.random.uniform(low=-1.5, high=-0.5, size=(50, 1))
    X4 = np.hstack((x4, y4))

    X = np.vstack((X1, X2, X3, X4))

    y = ((X[:, 1] * X[:, 0] < 0)).astype(int)
    # X, y = make_blobs(n_samples=100, centers=4, n_features=4,random_state=0)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

    plt.show()
