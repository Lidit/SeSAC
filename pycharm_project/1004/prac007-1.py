import matplotlib.pyplot as plt
import numpy as np

PI = np.pi
t = np.linspace(-4 * PI, 4 * PI, 1000)
sin = np.sin(t)
cos = np.cos(t)
tan = np.tan(t)
# boolean indexing
tan[:-1][np.diff(tan) < 0] = np.nan

fig, axes = plt.subplots(3, 1, figsize=(7, 10))

axes[0].plot(t, sin)
axes[1].plot(t, cos)
axes[2].plot(t, tan)

fig.tight_layout()

axes[2].set_ylim([-5, 5])

x_ticks = np.arange(-4 * PI, 4 * PI + 0.1, PI)
# LaTeX 수식 출력하기
x_ticklabels = [str(i) + r'$\pi$' for i in range(-4, 5)]

for i in range(len(axes)):
    axes[i].set_xticks(x_ticks)
    axes[i].set_xticklabels(x_ticklabels)
    axes[i].grid()

plt.show()
