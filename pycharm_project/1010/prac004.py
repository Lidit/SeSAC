# random centroid
import numpy as np
import matplotlib.pyplot as plt

centroid = np.random.uniform(-5, 5, (2,))

n_data = 100
data = np.random.normal(loc=centroid, scale=1, size=(n_data, 2))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(centroid[0], centroid[1], c='red')
ax.scatter(data[:, 0], data[:, 1])

plt.show()

# print(data[0:-1][1])
# x_data = np.random.normal(loc=centroid[0], scale=1, size=(100, ))
# y_data = np.random.normal(loc=centroid[1], scale=1, size=(100, ))
#
# fig, ax = plt.subplots(figsize=(10, 10))
#
# ax = plt.scatter(x=x_data,y=y_data)
# ax = plt.scatter(x=centroid[0], y=centroid[0], c='red')
#
# plt.show()
