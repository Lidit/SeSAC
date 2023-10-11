import numpy as np
import matplotlib.pyplot as plt

np.random.seed(15)

n_classes = 4
n_data = 100
x_data = []
y_data = []

color_list = ['blue', 'green', 'orange', 'red']

for class_idx in range(n_classes):
    centroid = np.random.uniform(low=-10, high=10, size=(2,))
    x_data_ = np.random.normal(loc=centroid, scale=1, size=(n_data, 2))
    y_data_ = np.ones((n_data,), dtype=int) * class_idx

    x_data.append(x_data_)
    y_data.append(y_data_)

x_data = np.vstack(x_data)
y_data = np.hstack(y_data)

print(x_data)
print(y_data.shape)

test_data = np.random.uniform(low=-10, high=10, size=(2,))

# for-loop 활용
# for data_idx in range(1, len(x_data)):
#     diff_square_sum = 0
#     for dim_idx in range(len(x_data[0])):
#         diff_square_sum += (x_data[0, dim_idx] - x_data[data_idx, dim_idx]) ** 2
#     e_distances.append(diff_square_sum ** 0.5)
# print(e_distances)

# broadcast 활용 euclidean distance 구하기
# e_distances = np.sqrt(np.sum((x_data - test_data) ** 2, axis=1))
# k = 5
# indices = np.argsort(e_distances)[:k]
# k_nearest_labels = y_data[indices]
# unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
# predicted_label = unique_labels[np.argmax(counts)]


def knn_classifier(t_data, data, k=5):
    e_distances = np.sqrt(np.sum((t_data - data) ** 2, axis=1))
    indices = np.argsort(e_distances)[:k]
    k_nearest_labels = y_data[indices]
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    predicted_label = unique_labels[np.argmax(counts)]

    return predicted_label


print("test_data class:", knn_classifier(test_data, x_data))

fig, ax = plt.subplots(figsize=(5, 5))

for class_idx in range(n_classes):
    data = x_data[class_idx * n_data: (class_idx + 1) * n_data]
    ax.scatter(x=data[:, 0], y=data[:, 1], alpha=0.5, color=color_list[class_idx])

ax.scatter(test_data[0], test_data[1], marker='*', s=100, c=color_list[knn_classifier(test_data, x_data)])
ax.text(test_data[0]+0.3, test_data[1]+0.3, 'class {}'.format(knn_classifier(test_data, x_data)),
        c='magenta', fontsize=10, ha='left')

x = np.linspace(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1, 100)
y = np.linspace(ax.get_ylim()[0] - 1, ax.get_ylim()[1] + 1, 100)
xx, yy = np.meshgrid(x, y)
mesh = np.c_[xx.ravel(), yy.ravel()]

grid_classes = []

for idx in range(len(mesh)):
    class_idx = int(knn_classifier(mesh[idx], x_data))
    grid_classes.append(color_list[class_idx])

# plot decision boundary
ax.scatter(mesh[:, 0], mesh[:, 1], alpha=0.1, color=grid_classes, s=2)

plt.show()
