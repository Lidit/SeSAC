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


#
# func1 = Function1()
# func2 = Function2()
#
# z = func1.forward(3)
# y = func2.forward(z)
#
# dy_dz = func2.backward()
# dz_dx = func1.backward(dy_dz=dy_dz)

# print(dz_dx)


def learn(x):
    func1 = Function1()
    func2 = Function2()

    z = func1.forward(x)
    y = func2.forward(z)

    dy_dz = func2.backward()
    dz_dx = func1.backward(dy_dz)

    print(f"x = {x}에서의 y = {y}, y' = {dz_dx}")


learn(5)
