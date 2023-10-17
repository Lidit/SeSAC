import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

n_data = 100
k = 2

color_list = ['blue', 'green', 'orange', 'red']

x_data = np.random.uniform(low=-10, high=10, size=(n_data, 2))


def e_distance(reference_point, point):
    return np.sqrt(np.sum((reference_point - point) ** 2))


print(x_data)
centroids = x_data[np.random.choice(len(x_data), k, replace=False)]
print(centroids)
# cnt = 0
# for _ in range(len(x_data)):
#     cnt += 1
#     distances = np.linalg.norm(x_data[:, np.newaxis, :] - centroids, axis=2)
#     # print(distances)
#     data_classes = np.argmin(distances, axis=1)
#     # print(data_classes)
#     new_centroids = np.array([x_data[data_classes == i].mean(axis=0) for i in range(k)])
#     # print(new_centroids)
#     if np.all(centroids == new_centroids):
#         break
#
#     centroids = new_centroids

distances = np.linalg.norm(x_data[:, np.newaxis, :] - centroids, axis=2)
# print(x_data[:, np.newaxis, :])
# print(x_data)
# print(distances)
# print(distances[0,1])
# print(distances.shape)
# data_classes = np.argmin(distances, axis=1)
# print(data_classes)
# print(centroids)
# print(x_data.shape)
# print(data_classes.shape)
# print(data_classes)

# plt.show()

data_classes = []

for point_idx in range(len(x_data)):
    cent1 = e_distance(centroids[0], x_data[point_idx])
    cent2 = e_distance(centroids[1], x_data[point_idx])

    if cent1 > cent2:
        data_classes.append(0)
    else:
        data_classes.append(1)
data_classes = np.hstack(data_classes)
print(data_classes)

# print(init_centroids[0])
# print(x_data[0])
# print(e_distance(init_centroids[0], x_data[0]))
# print(data_classes)
