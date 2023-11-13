import matplotlib.pyplot as plt
import numpy as np


def f(x1, x2): return 3 * x1 ** 2 + x2 ** 2


def df_dx1(x1): return 6 * x1


def df_dx2(x2): return 2 * x2


x1, x2 = 3, -2
ITERATIONS = 100
lr = 0.03
x1_track, x2_track, y_track = [x1], [x2], [f(x1, x2)]

for iter in range(ITERATIONS):
    dy_dx1 = df_dx1(x1)
    dy_dx2 = df_dx2(x2)
    x1 = x1 - (dy_dx1 * lr)
    x2 = x2 - (dy_dx2 * lr)

    x1_track.append(x1)
    x2_track.append(x2)
    y_track.append(f(x1, x2))

    print(f"{iter + 1} -th x1: {x1:.4f}")
    print(f"{iter + 1} -th x1: {x2:.4f}")
    print(f"{iter + 1} -th dy_x1: {dy_dx1:4f}")
    print(f"{iter + 1} -th dy_x2: {dy_dx2:4f}")

function_x1 = np.linspace(-5, 5, 100)
function_x2 = np.linspace(-5, 5, 100)

function_X1, function_X2 = np.meshgrid(function_x1, function_x2)
function_Y = np.log(f(function_X1, function_X2))

fig, ax = plt.subplots(figsize=(10, 10))

ax.contour(function_X1, function_X2, function_Y, levels=100, cmap="Reds_r", zorder=-1)

# X1_track, X2_track = np.meshgrid(x1_track, x2_track)
# Y_track = f(X1_track, X2_track)


ax.scatter(x1_track, x2_track, y_track, c=range(ITERATIONS + 1), cmap='rainbow')

ax.set_xlabel('x1', fontsize=15)
ax.set_ylabel('x2', fontsize=15)
ax.tick_params(labelsize=15)
fig.tight_layout()
plt.show()
