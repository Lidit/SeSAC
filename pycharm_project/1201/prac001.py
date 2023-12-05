import numpy as np
import matplotlib.pyplot as plt

N_SAMPLES = 50

x = np.random.uniform(-np.pi, np.pi, N_SAMPLES)
y = np.sin(x) + 0.2 * np.random.randn(N_SAMPLES)

fig, ax = plt.subplots(figsize=(10, 4))

ax.scatter(x, y)
ax.tick_params(labelsize=15)
ax.set_ylabel('y', fontsize=20)
ax.set_xlabel('x', fontsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.show()
