# knn x dataset
import numpy as np
import matplotlib.pyplot as plt

# my code
# color_list = ['red','blue','green','orange']
#
# fig, ax = plt.subplots(figsize=(5, 5))
#
# for color in color_list:
#     centroid = np.random.uniform(-5, 5, (2,))
#
#     n_data = 100
#     data = np.random.normal(loc=centroid, scale=0.5, size=(n_data, 2))
#
#     ax.scatter(data[:, 0], data[:, 1], c=color, alpha=0.3)
#     ax.scatter(centroid[0], centroid[1], marker='x', s=100, c='purple')
#
# plt.show()
#


n_classes = 4
n_data = 100
data = []
centroids = []

for _ in range(n_classes):
    centroid = np.random.uniform(low=-10, high=10, size=(2,))
    data_ = np.random.normal(loc=centroid, scale=1, size=(n_data, 2))

    centroids.append(centroid)
    data.append(data_)

centroids = np.vstack(centroids)
data = np.vstack(data)

fig, ax = plt.subplots(figsize=(5, 5))
for class_idx in range(n_classes):
    data_ = data[class_idx * n_data: (class_idx + 1) * n_data]

    ax.scatter(data_[:, 0], data_[:, 1], alpha=0.5)

for centroid in centroids:
    ax.scatter(centroid[0], centroid[1], c='purple', marker='x', s=100)

plt.show()