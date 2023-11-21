from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt

def get_dataset(N_SAMPLES, BATCH_SIZE):
    x, y = make_moons(n_samples=N_SAMPLES, noise=0.3)
    data = TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
    dataset = DataLoader(data, batch_size=BATCH_SIZE)

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
    fig, axes = plt.subplots(2,1, figsize=(10,5))

    axes[0].plot(losses)
    axes[1].plot(accs)

    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("BCE Loss", fontsize=15)
    axes[1].set_ylabel("Accuracy", fontsize=15)
    axes[0].tick_params(labelsize=10)
    axes[1].tick_params(labelsize=10)
    fig.tight_layout()
    plt.show()