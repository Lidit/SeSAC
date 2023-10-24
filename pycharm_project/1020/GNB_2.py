from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})
targets = df['target'].unique()
df.columns = df.columns.str.replace(' ', '_')
features = list(df.iloc[:, :4])
test_data = df.iloc[0]


def normalizator(mean, std, data):
    result = 1 / np.sqrt(2 * np.pi * np.square(std)) * (
            np.e ** (-0.5 * np.square((data - mean) / std)))

    return result


fig, axes = plt.subplots(4, 1, figsize=(7, 10))
likelihoods = []

for ax_idx, ax in enumerate(axes.flat):
    ax.set_xlim(df[features[ax_idx]].min()-0.1, df[features[ax_idx]].max()+0.1)
    likelihood = []
    for t_idx, target in enumerate(targets):
        mean = df[(df['target'] == target)].iloc[:, ax_idx].mean()
        std = df[(df['target'] == target)].iloc[:, ax_idx].std()
        x = np.linspace(df[features[ax_idx]].min(), df[features[ax_idx]].max(), 1000)
        y = np.array(normalizator(mean, std, x))
        ax.plot(x, y, label=target + " = " + str(round(normalizator(mean, std, test_data[features[ax_idx]]), 3)))
        ax.scatter(test_data[features[ax_idx]], normalizator(mean, std, test_data[features[ax_idx]]))
        likelihood.append(normalizator(mean, std, test_data[features[ax_idx]]))
        ax.set_title(features[ax_idx])
        ax.legend()
    likelihoods.append(likelihood)

plt.tight_layout()
plt.show()

posterior = [1, 1, 1]

for idx in range(len(likelihoods)):
    posterior[0] *= likelihoods[idx][0]
    posterior[1] *= likelihoods[idx][1]
    posterior[2] *= likelihoods[idx][2]

print('class of test_data: ', posterior.index(max(posterior)))
