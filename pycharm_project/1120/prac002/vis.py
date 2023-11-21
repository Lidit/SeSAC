import numpy as np
import matplotlib.pyplot as plt

import torch
from utils import get_xor_dataset

N_SAMPLES = 200
dataset = get_xor_dataset(N_SAMPLES)
X, y = dataset.tensors

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

model = torch.load('model.pt').to('cpu')

xlim, ylim = ax.get_xlim(), ax.get_ylim()
x1 = np.linspace(xlim[0], xlim[1], 100)
x2 = np.linspace(ylim[0], ylim[1], 100)
X1, X2 = np.meshgrid(x1, x2)
X_db = torch.FloatTensor(np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)]))
with torch.no_grad(): y_db = model(X_db)

X_db, y_db = X_db.numpy(), (y_db > 0.5).type(torch.float).numpy()

# X_db, y_db = X_db.numpy(), y_db.detach().numpy() # no grad를 안 했을 때, 모델 예측값만 가져오겠다
ax.scatter(X_db[:, 0], X_db[:, 1], c=y_db, cmap='bwr', alpha=0.1)
plt.show()
