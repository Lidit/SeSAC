import numpy as np


class Function1:
    def forward(self, x):
        z = x - 2
        self.dz_dx = 1

        return z

    def backward(self, dy_dz):
        dy_dx = self.dz_dx * dy_dz

        return dy_dx


class Function2:
    def forward(self, z):
        y = 2 * z ** 2

        # backward시 사용하기 위한 저장
        self.z = z

        return y

    def backward(self):
        dy_dz = 4 * self.z
        return dy_dz

class Function:
    def __init__(self):
        self.function1 = Function1()
        self.function2 = Function2()


    def forward(self,x):
        z = self.function1.forward(x)
        y = self.function2.forward(z)

        return z

    def backward(self):
        dy_dz = self.function2.backward()
        dy_dx = self.function1.backward(dy_dz)

        return dy_dx

