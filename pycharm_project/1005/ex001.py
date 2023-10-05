import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

feature_names = iris.feature_names
n_feature = len(feature_names)

species = iris.target_names
n_species = len(species)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

iris_X, iris_y = iris.data, iris.target

xticks = np.arange(3)

for ax_idx, ax in enumerate(axes.flat):
    setosa = np.array([])
    versicolor = np.array([])
    virginica = np.array([])
    for y_idx, y in enumerate(iris_y):
        if y == 0:
            setosa = np.append(setosa, iris_X[y_idx][ax_idx])
        elif y == 1:
            versicolor = np.append(versicolor, iris_X[y_idx][ax_idx])
        else:
            virginica = np.append(virginica, iris_X[y_idx][ax_idx])

    ax.violinplot([setosa, versicolor, virginica], positions=xticks)
    ax.set_title(feature_names[ax_idx], fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])

plt.tight_layout()

plt.show()
