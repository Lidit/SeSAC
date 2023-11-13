import numpy as np
import matplotlib.pyplot as plt


def f1(x): return 1 / 10 * x ** 2


def df_dx1(x): return 1 / 5 * x


def f2(x): return 1 / 8 * x ** 2


def df_dx2(x): return 1 / 4 * x


def f3(x): return 1 / 6 * x ** 2


def df_dx3(x): return 1 / 3 * x


x = [3, 3, 3]

ITERATIONS = 20
x_track1, y_track1 = [x[0]], [f1(x[0])]
x_track2, y_track2 = [x[1]], [f2(x[1])]
x_track3, y_track3 = [x[2]], [f3(x[2])]

# print(f'Initial x:{x}')

for iter in range(ITERATIONS):
    dy_dx1 = df_dx1(x[0])
    x[0] = x[0] - dy_dx1

    dy_dx2 = df_dx2(x[1])
    x[1] = x[1] - dy_dx2

    dy_dx3 = df_dx3(x[2])
    x[2] = x[2] - dy_dx3

    x_track1.append(x[0])
    y_track1.append(f1(x[0]))

    x_track2.append(x[1])
    y_track2.append(f2(x[1]))

    x_track3.append(x[2])
    y_track3.append(f3(x[2]))
    print(f"{iter + 1} -th x: {x[0]:.4f}")
    print(f"{iter + 1} -th x: {x[1]:.4f}")
    print(f"{iter + 1} -th x: {x[2]:.4f}")

fig, ax = plt.subplots(3, 1, figsize=(10, 10))

function1_x = np.linspace(-5, 5, 100)
function1_y = f1(function1_x)

function2_x = np.linspace(-5, 5, 100)
function2_y = f2(function1_x)

function3_x = np.linspace(-5, 5, 100)
function3_y = f3(function3_x)

ax[0].plot(function1_x, function1_y)
ax[0].scatter(x_track1, y_track1, c=range(ITERATIONS + 1), cmap='rainbow')

ax[1].plot(function2_x, function2_y)
ax[1].scatter(x_track2, y_track2, c=range(ITERATIONS + 1), cmap='rainbow')

ax[2].plot(function3_x, function3_y)
ax[2].scatter(x_track3, y_track3, c=range(ITERATIONS + 1), cmap='rainbow')

fig.tight_layout()

plt.show()
