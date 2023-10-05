import matplotlib.pyplot as plt
import numpy as np

# Size array and Color Array

n_data = 10
x_data = np.linspace(0, 10, n_data)
y_data = np.linspace(0, 10, n_data)

c_arr = [(c / n_data, c / n_data, c / n_data) for c in range(n_data)]
# c_arr = [(c/n_data, c/n_data, c/n_data) for c in range(n_data, 0, -1)] # 색 연한거 먼저

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_data, y_data, s=500, c=c_arr)

plt.show()
