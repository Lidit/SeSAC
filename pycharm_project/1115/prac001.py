import numpy as np
import matplotlib.pyplot as plt


class BCELoss:
    def forward(self, y, pred):
        self.y = y
        self.pred = pred
        j = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        return j

    def backward(self):
        dj_dpred = (self.pred - self.y) / self.pred * (1 - self.pred)

        return dj_dpred


class Sigmoid:
    def forward(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, dj_dpred):
        da_dz = self.a * (1 - self.a)
        # print(type(da_dz))
        # print(type(dj_dpred))
        dj_dz = dj_dpred * da_dz

        return dj_dz


class AffineFunction:
    def __init__(self):
        self.w = np.random.randn(2)
        self.b = np.random.randn(1)

    def forward(self, x):
        self.x = x
        z = np.dot(self.x, self.w) + self.b
        return z

    def backward(self, dj_dz, lr):
        dz_dw = self.x
        dz_db = 1

        dj_dw = dj_dz * dz_dw
        dj_db = dj_dz * dz_db

        self.w -= dj_dw * lr
        self.b -= dj_db * lr

        dz_dx = self.w
        dj_dx = dj_dz * dz_dx

        return dj_dx


class Model:
    def __init__(self):
        self.affine = AffineFunction()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.affine.forward(x)
        a = self.sigmoid.forward(z)

        return a

    def backward(self, dj_dpred, lr):
        dj_dz = self.sigmoid.backward(dj_dpred)
        test = self.affine.backward(dj_dz, lr)
        return test


LR = 0.1
EPOCHS = 10000
np.random.seed(7)
X = np.array([0, 0, 1, 0, 0, 1, 1, 1]).reshape(4, 2)
y = np.array([0, 1, 1, 0])
print(X)
print(y)
# Instantiation
neuron1 = Model()
neuron2 = Model()
neuron3 = Model()
loss_function = BCELoss()
bce_losses = []
accuracies = []
x1 = np.linspace(-0.5, 1.5, 100)
x2 = np.linspace(-0.5, 1.5, 100)
X1, X2 = np.meshgrid(x1, x2)
X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])


fig, axes = plt.subplots(4, 4, figsize=(10, 5))
axes = axes.flatten()
for epoch in range(1, EPOCHS + 1):
    epoch_losses = []
    epoch_accuracy = []
    # print("epoch: ", epoch)
    # vs_binary_pred = []
    for X_, y_ in zip(X, y):
        # training
        pred1 = neuron1.forward(X_)
        pred2 = neuron2.forward(X_)
        pred3 = neuron3.forward([pred1[0], pred2[0]])

        # get loss
        loss = loss_function.forward(y_, pred3)
        epoch_losses.append(loss)

        # backward
        dj_dpred3 = loss_function.backward()
        dj_dpred_back = neuron3.backward(dj_dpred3, LR)

        neuron2.backward(dj_dpred_back[1], LR)
        neuron1.backward(dj_dpred_back[0], LR)

        # Metric(loss, accuracy) Calculations
        pred_binary = (pred3 >= 0.5).astype(int) # 예측값을 이진 변환. 예측값이 0.5 이상이면 1, 0.5 미만이면 0으로
        epoch_accuracy.append(abs(pred_binary - y_))  # 이진변환된 값이 정답과 같다면 0, 다르다면 1이 epoch_accuracy list에 append된다.

    accuracies.append(epoch_accuracy.count(0) / len(y))
    bce_losses.append(np.mean(epoch_losses))

    if epoch % 625 == 0 and epoch > 1:
        y_db = []
        for x_db in X_db:
            p1 = neuron1.forward(x_db)
            p2 = neuron2.forward(x_db)
            # 주어진 x_db에 따른 최종 예측 결과를 y_db 에 append
            y_db.append(neuron3.forward([p1[0], p2[0]]))
        axes[(epoch // 625) - 1].scatter(X_db[:, 0], X_db[:, 1], c=y_db, cmap='bwr')
        print("epoch: ", epoch)

plt.show()
