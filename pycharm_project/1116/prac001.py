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
    def __init__(self, n=2):
        self.w = np.random.randn(n)
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
    def __init__(self, n=2):
        self.affine = AffineFunction(n)
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
EPOCHS = 7000

neuron1 = Model(2)
neuron2 = Model(2)
neuron3 = Model(2)
neuron4 = Model(3)
loss_function = BCELoss()
bce_losses = []
accuracies = []


# 데이터셋 생성 함수
def generate_nonlinear_data_triangle(num_samples):
    np.random.seed(42)
    X = np.random.uniform(low=-2, high=2, size=(num_samples, 2))
    # 삼각형 내부의 점들은 1, 외부의 점들은 0으로 레이블링
    y = ((X[:, 1] < X[:, 0] + 1) & (X[:, 1] < -X[:, 0] + 1) & (X[:, 1] > -1)).astype(int)

    return X, y


# 데이터셋 생성
num_samples = 500
X, y = generate_nonlinear_data_triangle(num_samples)

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
X_db = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
# x1 = np.linspace(-1.5, 1.5, 100)
# x2 = np.linspace(-1.5, 1.5, 100)
# X1, X2 = np.meshgrid(x1, x2)
# train_data = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
# y = []
#
# for data in train_data:
#     if data[0] < 0 <= data[1]:
#         y.append(1)
#     else:
#         y.append(0)
#
# y = np.array(y)

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
        pred3 = neuron3.forward(X_)
        pred4 = neuron4.forward([pred1[0], pred2[0], pred3[0]])
        # get loss
        loss = loss_function.forward(y_, pred4)
        epoch_losses.append(loss)

        # backward
        dj_dpred4 = loss_function.backward()
        dj_dpred_back = neuron4.backward(dj_dpred4, LR)

        neuron3.backward(dj_dpred_back[2], LR)
        neuron2.backward(dj_dpred_back[1], LR)
        neuron1.backward(dj_dpred_back[0], LR)

        # Metric(loss, accuracy) Calculations
        pred_binary = (pred3 >= 0.5).astype(int)  # 예측값을 이진 변환. 예측값이 0.5 이상이면 1, 0.5 미만이면 0으로
        epoch_accuracy.append(abs(pred_binary - y_))

    accuracies.append(epoch_accuracy.count(0) / len(y))
    bce_losses.append(np.mean(epoch_losses))

    if epoch % 500 == 0 and epoch > 1:
        y_db = []
        for x_db in X_db:
            p1 = neuron1.forward(x_db)
            p2 = neuron2.forward(x_db)
            p3 = neuron3.forward(x_db)
            # 주어진 x_db에 따른 최종 예측 결과를 y_db 에 append
            y_db.append(neuron4.forward([p1[0], p2[0], p3[0]]))
        axes[(epoch // 500) - 1].scatter(X_db[:, 0], X_db[:, 1], c=y_db, cmap='bwr')
        print("epoch: ", epoch)

plt.show()
