import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

n_data = 100
x_data = np.random.normal(0, 1, (n_data,))
y_data = np.random.normal(0, 1, (n_data,))

fig, ax = plt.subplots(figsize=(5, 5))

# ax.scatter(x_data, y_data, s=300, facecolor='white', edgecolor='tab:blue', linewidth=5)
# ax.scatter(x_data, y_data, s=300, facecolor='none', edgecolor='tab:blue', linewidth=5)
ax.scatter(x_data, y_data, s=300, facecolor='none', edgecolor='tab:blue', linewidth=5, alpha=0.5)

fig.show()
