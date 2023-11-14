import numpy as np


class Function1:
    def forward(self, x):
        z1 = -x


        return z1

    def backward(self, dy_dz2):
        dz_dx = -1
        dy_dx = dz_dx * dy_dz2

        return dy_dx


class Function2:

    def forward(self, z1):
        z2 = np.exp(z1)

        self.z1 = z1

        return z2

    def backward(self, dy_dz3):
        dz2_dz1 = np.exp(self.z1)
        dy_dz2 = dz2_dz1 * dy_dz3

        return dy_dz2


class Function3:
    def forward(self, z2):
        z3 = z2 + 1


        return z3

    def backward(self, dy_da):
        dz3_dz2 = 1
        dy_dz3 = dz3_dz2 * dy_da

        return dy_dz3


class Function4:
    def forward(self, z3):
        a = 1 / z3

        self.z3 = z3

        return a

    def backward(self):
        dy_da = - 1 / (self.z3 ** 2)

        return dy_da


class Sigmoid:
    def __init__(self):
        self.function1 = Function1()
        self.function2 = Function2()
        self.function3 = Function3()
        self.function4 = Function4()

    def forward(self, x):
        z1 = self.function1.forward(x)
        z2 = self.function2.forward(z1)
        z3 = self.function3.forward(z2)
        y = self.function4.forward(z3)

        return y

    def backward(self):
        dy_da = self.function4.backward()
        dy_dz3 = self.function3.backward(dy_da)
        dy_dz2 = self.function2.backward(dy_dz3)
        dy_dz1 = self.function1.backward(dy_dz2)

        return dy_dz1


if __name__ == '__main__':

    sigmoid = Sigmoid()

    print(sigmoid.forward(0))
    print(sigmoid.backward())
