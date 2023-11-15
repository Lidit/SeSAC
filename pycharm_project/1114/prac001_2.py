import numpy as np
from prac001_1 import BCELoss, Model
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(7)
N_SAMPLES = 100
LR = 0.001
EPOCHS = 30

X, y = make_blobs(n_samples=N_SAMPLES, centers=2, n_features=2, cluster_std=0.5, random_state=0)

# Instantiation
model = Model()
loss_function = BCELoss()
bce_losses = []
accuracies = []

for epoch in range(EPOCHS):
    epoch_losses = []
    epoch_accuracy = []
    for X_, y_ in zip(X, y):

        # training
        pred = model.forward(X_)
        loss = loss_function.forward(y_, pred)
        print(f"y = {y_}, pred = {pred}")
        print(f"loss={loss}")
        model.backward(loss_function.backward(), lr=LR)

        # Methric(loss, accuracy) Calcuations
        epoch_losses.append(loss[0])

        pred_binary = (pred >= 0.5).astype(int)
        # if pred_binary == y_:
        #     epoch_accuracy.append(1)
        #     print("accuracy = 1")
        # else:
        #     epoch_accuracy.append(0)
        #     print("accuracy = 0")
        epoch_accuracy.append(abs(pred_binary - y_))

    accuracies.append(epoch_accuracy.count(0) / len(y))
    bce_losses.append(np.mean(epoch_losses))

print(bce_losses)
print(accuracies)

# Result Visualization
fig, ax = plt.subplots(2, 1, figsize=(10, 5))

ax[0].set_ylabel("BECLoss", fontsize=15)
ax[0].plot(range(EPOCHS), bce_losses)

ax[1].set_ylabel("Accuracy", fontsize=15)
ax[1].set_xlabel('Epoch', fontsize=15)
ax[1].plot(range(EPOCHS), accuracies)

fig.tight_layout()
plt.show()
