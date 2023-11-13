import numpy as np


class AffineFunction:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        z = np.dot(self.w, x) + self.b

        return z


class Sigmoid:
    def forward(self, z):
        a = 1 / (1 + np.exp(-z))

        return a


class ArtificialNeuron:
    def __init__(self, w, b):
        self.affine = AffineFunction(w=w, b=b)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.affine.forward(x)
        a = self.sigmoid.forward(z)

        return a


class Model:
    def __init__(self):
        self.and_ = ArtificialNeuron(w=[0.5, 0.5], b=-0.7)
        self.or_ = ArtificialNeuron(w=[0.5, 0.5], b=-0.2)
        self.nand = ArtificialNeuron(w=[-0.5, -0.5], b=0.7)

    def forward(self, x):
        a1 = self.and_.forward(x)
        a2 = self.or_.forward(x)
        a3 = self.nand.forward(x)

        a = np.array([a1, a2, a3])
        return a


model = Model()

print(model.forward([1, 1]))
