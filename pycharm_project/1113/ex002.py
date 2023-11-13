# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def f(x): return 1 / 10 * x ** 2
#
#
# def df_dx(x): return 1 / 5 * x
#
#
# def f_(x): return 1 / 10 * ((x - 2) ** 2)
#
#
# def df_dx_(x): return 1 / 5 * (x - 2)
#
#
# x = 3
# ITERATIONS = 20
#
# x_list = np.linspace(-5, 5, 1000)
#
# print(f'Initial x:{x}')
#
# fig, ax = plt.subplots(2, 1, figsize=(10, 5))
#
# ax[0].plot(x_list, f(x_list))
#
# for iter in range(ITERATIONS):
#     dy_dx = df_dx(x)
#     x = x - dy_dx
#     ax[0].scatter(x, f(x))
#     print(f"{iter + 1} -th x: {x:.4f}")
#
# x_list2 = np.linspace(0, 20, 1000)
#
# ax[1].plot(x_list2, f_(x_list2))
# x_ = 0
# for iter in range(ITERATIONS):
#     dy_dx = df_dx_(x)
#     x = x - dy_dx
#     ax[1].scatter(x, f_(x))
#     print(f"{iter + 1} -th x: {x:.4f}")
#
# plt.show()
import numpy as np
import matplotlib.pyplot as plt


def f(x): return 1 / 10 * x ** 2


def df_dx(x): return 1 / 5 * x


x = 3
ITERATIONS = 20
x_track, y_track = [x], [f(x)]
print(f'Initial x:{x}')

for iter in range(ITERATIONS):
    dy_dx = df_dx(x)
    x = x - dy_dx

    x_track.append(x)
    y_track.append(f(x))
    print(f"{iter + 1} -th x: {x:.4f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
function_x = np.linspace(-5, 5, 100)
function_y = f(function_x)

ax[0].plot(function_x, function_y)
ax[0].scatter(x_track, y_track, c=range(ITERATIONS + 1), cmap='rainbow')
ax[0].set_xlabel('x', fontsize=15)
ax[0].set_ylabel('y', fontsize=15)

ax[1].plot(x_track, marker='o')
ax[1].set_xlabel('Iteration', fontsize=15)
ax[1].set_ylabel('x', fontsize=15)
fig.tight_layout()

plt.show()
