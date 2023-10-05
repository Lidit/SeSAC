import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

feature_names = iris.feature_names
n_feature = len(feature_names)
species = iris.target_names
n_species = len(species)

fig, axes = plt.subplots(n_feature, n_feature, figsize=(10, 10))

iris_X, iris_y = iris.data, iris.target

xticks = np.arange(3)

feature_list = []

for feature_idx in range(n_feature):
    feature_data = []
    for y_idx, y in enumerate(iris_y):
        feature_data.append(iris_X[y_idx][feature_idx])
    feature_list.append(feature_data)

feature_list = np.array(feature_list)

for ax_idx, ax in enumerate(axes):
    ax[0].set_ylabel(feature_names[ax_idx], fontsize=15)
    for sub_ax_idx, sub_ax in enumerate(ax.flat):
        if ax_idx == sub_ax_idx:
            sub_ax.hist(feature_list[sub_ax_idx], rwidth=0.9)
        else:
            sub_ax.scatter(feature_list[sub_ax_idx][:50], feature_list[ax_idx][:50], c='purple', alpha=0.6)
            sub_ax.scatter(feature_list[sub_ax_idx][50:101], feature_list[ax_idx][50:101], c='green', alpha=0.6)
            sub_ax.scatter(feature_list[sub_ax_idx][100:], feature_list[ax_idx][100:], c='yellow', alpha=0.6)
        if ax_idx == 3:
            sub_ax.set_xlabel(feature_names[sub_ax_idx], fontsize=15)

plt.tight_layout()

plt.show()


## gpt코드

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import numpy as np
#
# iris = load_iris()
#
# feature_names = iris.feature_names
# n_feature = len(feature_names)
# species = iris.target_names
#
# fig, axes = plt.subplots(n_feature, n_feature, figsize=(10, 10))
#
# iris_X, iris_y = iris.data, iris.target
#
# for ax_idx, ax in enumerate(axes):
#     ax[0].set_ylabel(feature_names[ax_idx], fontsize=15)
#     for sub_ax_idx, sub_ax in enumerate(ax.flat):
#         if ax_idx == sub_ax_idx:
#             sub_ax.hist(iris_X[:, ax_idx], rwidth=0.9)
#         else:
#             for class_idx, color in enumerate(['purple', 'green', 'yellow']):
#                 subset = iris_X[iris_y == class_idx]
#                 sub_ax.scatter(
#                     subset[:, sub_ax_idx],
#                     subset[:, ax_idx],
#                     c=color,
#                     alpha=0.6,
#                     label=species[class_idx]
#                 )
#         if ax_idx == n_feature - 1:
#             sub_ax.set_xlabel(feature_names[sub_ax_idx], fontsize=15)
#
# plt.tight_layout()
# plt.show()

## gpt코드

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# import numpy as np
#
# iris = load_iris()
#
# feature_names = iris.feature_names
# species = iris.target_names
#
# fig, axes = plt.subplots(4, 4, figsize=(12, 12))
#
# iris_X, iris_y = iris.data, iris.target
#
# colors = ['purple', 'green', 'yellow']
#
# for ax_idx, ax in enumerate(axes):
#     ax[0].set_ylabel(feature_names[ax_idx], fontsize=15)
#     for sub_ax_idx, sub_ax in enumerate(ax):
#         if ax_idx == sub_ax_idx:
#             sub_ax.hist(iris_X[:, ax_idx], rwidth=0.9)
#         else:
#             for class_idx, color in enumerate(colors):
#                 subset = iris_X[iris_y == class_idx]
#                 sub_ax.scatter(
#                     subset[:, sub_ax_idx],
#                     subset[:, ax_idx],
#                     c=color,
#                     alpha=0.6,
#                     label=species[class_idx]
#                 )
#         if ax_idx == 3:
#             sub_ax.set_xlabel(feature_names[sub_ax_idx], fontsize=15)
#
# plt.tight_layout()
# plt.show()
