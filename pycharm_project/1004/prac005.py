import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

x_data = np.array([10, 25, 31, 40, 55, 80, 100])
y_data = np.random.normal(0, 1, (7,))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_data, y_data)

fig.subplots_adjust(left=0.2)
ax.tick_params(labelsize=25)

ax.set_xticks(x_data)
ylim = ax.get_ylim()
yticks = np.linspace(ylim[0], ylim[1], 8)
ax.set_yticks(yticks)

plt.show()
