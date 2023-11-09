import numpy as np
import matplotlib.pyplot as plt


class AffineFunction:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        x = np.array(x)

        # return np.sum(x * self.w) + self.b
        return np.dot(x, self.w) + self.b


class Sigmoid:
    def forward(self, z):
        z = np.array(z)
        y = 1 / (1 + np.exp(-z))

        return y


affine1 = AffineFunction(w=[1, 1], b=-1.5)
affine2 = AffineFunction(w=[-1, -1], b=0.5)

# print(f"affine1 w:{affine1.w}, b:{affine1.b}")
# print(f"affine2 w:{affine2.w}, b:{affine2.b}")

sigmoid = Sigmoid()

print(sigmoid.forward([-2.5, -1, 0, 1, 2.5]))

fig, ax = plt.subplots(figsize=(10, 5))

x = np.linspace(-5, 5, 1000)

ax.axvline(0, color='black', linewidth=1, alpha=0.5)
ax.axhline(0, color='black', linewidth=1, alpha=0.5)
ax.plot(x, sigmoid.forward(x))

plt.show()
