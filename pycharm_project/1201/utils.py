import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn as nn

np.random.seed(8)


def save_losses_to_csv(losses, save_path):
    losses = pd.Series(name='losses', data=losses)
    losses.to_csv(os.path.join(save_path, 'result.csv'))


def get_sin_ds(n_samples):
    x = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.sin(x) + 0.2 * np.random.randn(n_samples)
    return x, y


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    return device


def train(data, n_samples, model, loss_function, optimizer, DEVICE):
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

    epoch_loss /= n_samples
    epoch_accuracy = epoch_corrects / n_samples
    return epoch_loss, epoch_accuracy


# def vis_losses_accs(losses, accs):
#     fig, axes = plt.subplots(2, 1, figsize=(10, 5))
#
#     axes[0].plot(losses)
#     axes[1].plot(accs)
#
#     axes[1].set_xlabel("Epoch", fontsize=15)
#     axes[0].set_ylabel("Loss", fontsize=15)
#     axes[1].set_ylabel("Accuracy", fontsize=15)
#
#     axes[0].tick_params(labelsize=10)
#     axes[1].tick_params(labelsize=10)
#
#     fig.tight_layout()
#     plt.show()

def vis_losses_pred(x, y, losses, model, device, save_path='result'):
    x_model = torch.linspace(x.min(), x.max(), 100).view(100, 1).to(device)
    y_model = model(x_model)
    x_model, y_model = x_model.to('cpu'), y_model.detach().to('cpu')

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].plot(losses, label='train loss')
    axes[0].legend(fontsize=15)
    axes[0].set_xlabel('Epoch', fontsize=15)
    axes[0].set_ylabel('MSE Loss', fontsize=15)

    axes[1].scatter(x, y, label='ds')
    axes[1].plot(x_model, y_model, label='model', color='red', lw=3)
    axes[1].legend(fontsize=15)
    axes[1].tick_params(labelsize=15)
    axes[1].set_xlabel('X', fontsize=20)
    axes[1].set_ylabel('Y', fontsize=20)
    fig.tight_layout()

    for spine_loc, spine in axes[1].spines.items():
        if spine_loc in ['right', 'top']:
            spine.set_visible(False)
    fig.savefig(os.path.join(save_path, 'result.png'))


def random_split(x, y, train_val_ratio):
    n_total_samples = len(x)

    n_train_samples = int(train_val_ratio * n_total_samples)
    n_val_samples = n_total_samples - n_train_samples

    random_idx = np.arange(n_total_samples)
    np.random.shuffle(random_idx)
    train_idx = random_idx[:n_train_samples]
    val_idx = random_idx[n_train_samples:]

    train_x, train_y = x[train_idx], y[train_idx]
    val_x, val_y = x[val_idx], y[val_idx]
    return train_x, train_y, val_x, val_y, n_train_samples, n_val_samples
