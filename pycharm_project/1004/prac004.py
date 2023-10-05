import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

n_data = 100
s_idx = 30
x_data = np.arange(s_idx, s_idx + n_data )
y_data = np.random.normal(0, 1, (n_data, ))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_data, y_data)

fig.tight_layout(pad=3)
x_ticks = np.arange(s_idx, s_idx + n_data + 1, 20)
ax.set_xticks(x_ticks)

ax.tick_params(labelsize=25)
ax.grid()

plt.show()
