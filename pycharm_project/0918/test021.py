import matplotlib.pyplot as plt
import numpy as np

# Size array and Color Array
np.random.seed(0)

n_data = 500
x_data = np.random.normal(0, 1, size=(n_data,))
y_data = np.random.normal(0, 1, size=(n_data,))
s_arr = np.random.uniform(100, 500, n_data)
c_arr = [np.random.uniform(0, 1, 3) for _ in range(n_data)]

fig, ax = plt.subplots(figsize=(10, 10))

# ax.scatter(x_data, y_data, s=500, c=c_arr)
ax.scatter(x_data, y_data, s=500, c=c_arr, alpha=0.3)

plt.show()
