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
        dj_dx = self.affine.backward(dj_dz, lr)
        return dj_dx


LR = 0.1
EPOCHS = 10000

neuron1 = Model()
neuron2 = Model()
neuron3 = Model()
neuron4 = Model()
loss_function = BCELoss()

x1 = np.array([])
x2 = np.array([])
mu = np.array([])
y = []
train_data = np.hstack([x1.reshape(1,-1),x2.reshape(1,-1),mu.reshape(1,-1)])
for epoch in range(1, EPOCHS + 1):
    epoch_losses = []
    epoch_accuracy = []
    # print("epoch: ", epoch)
    # vs_binary_pred = []
    for X_, y_ in zip(x1, y):
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
        epoch_accuracy.append(abs(pred_binary - y_))