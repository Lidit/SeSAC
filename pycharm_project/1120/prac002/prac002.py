import numpy as np
from utils import get_device, get_dataset, train, vis_losses_accs, vis_decision_boundary
from model import MLP
import torch.nn as nn
from torch.optim import SGD
import torch

N_SAMPLES = 300
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.1
DEVICE = get_device()

dataloader = get_dataset(N_SAMPLES=N_SAMPLES, BATCH_SIZE=BATCH_SIZE)

model = MLP().to(DEVICE)
loss_function = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=LR)
losses, accs = [], []

for epoch in range(EPOCHS):
    epoch_loss, epoch_acc = train(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE)

    losses.append(epoch_loss)
    accs.append(epoch_acc)

    print(f"Epoch: {epoch + 1}")
    print(f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}\n")

vis_losses_accs(losses, accs)

vis_decision_boundary(x=dataloader.dataset[:][0], model=model, DEVICE=DEVICE)
# model_scripted = torch.jit.script(model) # TorchScript 형식으로 내보내기
# model_scripted.save('./model.pt')
