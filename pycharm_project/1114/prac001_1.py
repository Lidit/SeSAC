import numpy as np




class BCELoss:
    def forward(self, y, pred):
        j = -( y * np.log(pred) + ( (1 - y) * np.log(1 - pred) ) )
        self.y = y
        self.y_pred = pred
        return j

    def backward(self):
        dj_dy_pred = (self.y_pred - self.y) / self.y_pred * (1 - self.y_pred)

        return dj_dy_pred


class Sigmoid:
    def forward(self, z):
        # z = np.array(z)
        self.y_pred = 1 / (1 + np.exp(-z))
        return self.y_pred

    def backward(self, dj_dy_pred):
        dy_pred_dz = self.y_pred * (1 - self.y_pred)
        dj_dz = dy_pred_dz * dj_dy_pred

        return dj_dz


class AffineFunction:
    def __init__(self):
        self.w = np.random.randn(2)
        self.b = np.random.randn(1)

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.w) + self.b
        return z

    def backward(self, dy_pred_dz, lr):
        dz_dw = self.x
        dz_db = 1

        dj_dw = dy_pred_dz * dz_dw
        dj_db = dy_pred_dz * dz_db

        self.w -= (dj_dw * lr)
        self.b -= dj_db * lr


class Model:
    def __init__(self):
        self.affine = AffineFunction()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.affine.forward(x)
        a = self.sigmoid.forward(z)

        return a

    def backward(self, dj_dy_pred, lr):
        dj_dz = self.sigmoid.backward(dj_dy_pred)
        # print(f"갱신전 w: {self.affine.w}")
        # print(f"갱신전 b: {self.affine.b}")
        self.affine.backward(dj_dz, lr)
        print(f"갱신된 w: {self.affine.w}")
        print(f"갱신된 b: {self.affine.b}")

if __name__ == '__main__':
    np.random.seed(0)

    model = Model()
    loss_function = BCELoss()
    y = 1

    pred = model.forward([1, 1])
    loss = loss_function.forward(y=y, pred=pred)
    print(f"y = 1, pred = {pred}")
    print(f"loss={loss}")
    model.backward(loss_function.backward(), lr=0.3)
