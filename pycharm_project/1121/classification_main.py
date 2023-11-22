import torch.nn as nn
from sklearn.datasets import make_blobs
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Classifier
from utils import get_classification_dataset, get_device, train_classification, vis_decision_boundary

DEVICE = get_device()

# hyper parameter
EPOCHS = 50
N_SAMPLES = 200
BATCH_SIZE = 8
LR = 0.1

# 데이터 생성
X, y = make_blobs(n_samples=N_SAMPLES, n_features=2, cluster_std=0.7, centers=4)
dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# 생성된 데이터 plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1')
plt.show()

# 모델 인스턴스 생성
model = Classifier().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)

losses, accs = [], []
# 학습
for epoch in tqdm(range(EPOCHS), "Train in progress"):
    epoch_loss, epoch_acc = train_classification(dataloader, N_SAMPLES, model, loss_function, optimizer, DEVICE)

    losses.append(epoch_loss)
    accs.append(epoch_acc)

    print(f"Epoch: {epoch + 1}")
    print(f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}\n")

# 모델 학습 이후에, 학습 데이터에 대한 예측값 plot
vis_decision_boundary(dataloader=dataloader, DEVICE=DEVICE, model=model)
