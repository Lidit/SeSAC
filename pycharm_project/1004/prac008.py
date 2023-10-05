import matplotlib.pyplot as plt
import numpy as np

PI = np.pi
t = np.linspace(-4 * PI, 4 * PI, 1000)
t_reshape = t.reshape(1, -1)

print(t)
print(t_reshape)
print("t shape:", t.shape)
print("t_reshape shape: ", t_reshape.shape)

sin = np.sin(t)
cos = np.cos(t)
tan = np.tan(t)
# boolean indexing for tan
tan[:-1][np.diff(tan) < 0] = np.nan

# vstack -> vertical stack
# hstack -> horizontal stack
data = np.vstack((sin, cos, tan))
print("data shape: ", data.shape)
print(data)

title_list = [r'$sin(t)$', r'$cos(t)$', r'$tan(t)$']
x_ticks = np.arange(-4 * PI, 4 * PI + PI, PI)
x_ticklabels = [str(i) + r'$\pi$' for i in range(-4, 5)]

fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

for ax_idx, ax in enumerate(axes.flat):
    ax.plot(t_reshape.flatten(), data[ax_idx])
    ax.set_title(title_list[ax_idx], fontsize=30)
    ax.tick_params(labelsize=20)
    ax.grid()
    if ax_idx == 2:
        ax.set_ylim([-3, 3])

fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95)

axes[-1].set_xticks(x_ticks)
axes[-1].set_xticklabels(x_ticklabels)

print(type(axes))
print(axes.shape)
print(np.hstack((sin, cos, tan)))
print('\n')
print(np.vstack((sin, cos, tan)))
plt.show()
