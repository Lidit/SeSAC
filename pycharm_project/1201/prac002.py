import torch
from tqdm import tqdm
import numpy as np
import os
import torch.nn as nn
from torch import FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, lr_scheduler

from model import SinRegressor
from utils import get_device, get_sin_ds, vis_losses_pred, save_losses_to_csv, random_split

N_SAMPLES = 100
TRAIN_VAL_RATIO = 0.8

BATCH_SIZE = 2
LR = 0.1
EPOCHS = 3000
LR_DECAY_RATE = 0.9999
# SAVE_PATH = 'result'
SAVE_PATH = 'result_val'

x, y = get_sin_ds(N_SAMPLES)
train_x, train_y, val_x, val_y, n_train_samples, n_val_samples = random_split(x, y, TRAIN_VAL_RATIO)

train_ds = TensorDataset(FloatTensor(train_x), FloatTensor(train_y))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
# dataset = TensorDataset(FloatTensor(x), FloatTensor(y))

val_ds = TensorDataset(FloatTensor(val_x), FloatTensor(val_y))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = get_device()

model = SinRegressor().to(device)
loss_function = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=LR)
scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: LR_DECAY_RATE ** epoch, verbose=False)

# losses = []
train_losses, val_losses = [], []
for epoch in tqdm(range(EPOCHS)):
    # epoch_loss = 0
    train_loss_epoch = 0
    for x_, y_ in train_loader:
        x_, y_ = x_.to(device), y_.to(device)
        x_ = x_.view(x_.size(0), 1)

        pred = model(x_).flatten()
        loss = loss_function(pred, y_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * len(x_)
    scheduler.step()

    train_loss_epoch /= N_SAMPLES
    train_losses.append(train_loss_epoch)

    val_loss_epoch = 0
    with torch.no_grad():
        for x_, y_ in val_loader:
            x_, y_ = x_.to(device), y_.to(device)
            x_ = x_.view(x_.size(0), 1)

            pred = model(x_).flatten()
            loss = loss_function(pred, y_)

            val_loss_epoch += loss.item() * len(x_)

        val_loss_epoch /= N_SAMPLES
        val_losses.append(val_loss_epoch)

os.makedirs(SAVE_PATH, exist_ok=True)
# vis_losses_pred(x, y, losses, model, device, save_path=SAVE_PATH)
# save_losses_to_csv(losses, SAVE_PATH)
vis_losses_pred(x, y, train_losses, model, device, save_path=SAVE_PATH)
save_losses_to_csv(train_losses, SAVE_PATH)
