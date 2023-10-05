import matplotlib.pyplot as plt
import numpy as np

PI = np.pi
t = np.linspace(-4 * PI, 4 * PI, 1000)
sin = np.sin(t)
cos = np.cos(t)
tan = np.tan(t)

fig, axes = plt.subplots(3, 1, figsize=(7, 10))

axes[0].plot(t, sin)
axes[1].plot(t, cos)
axes[2].plot(t, tan)

fig.tight_layout()

axes[2].set_ylim([-5, 5])

plt.show()
