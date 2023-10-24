from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# df['target'] = df['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})
targets = ["setosa", "versicolor","virginica"]
df.columns = df.columns.str.replace(' ', '_')
# print(df)

test_data = df.iloc[0]


def normalizator(mean, std, feature, data):
    result = 1 / np.sqrt(2 * np.pi * np.square(std[feature])) * (
            np.e ** (-0.5 * np.square((data[feature] - mean[feature]) / std[feature])))

    return result


setosa_mean = df[(df['target'] == 0)].iloc[:, :4].mean()
setosa_std = df[(df['target'] == 0)].iloc[:, :4].std()

versicolor_mean = df[(df['target'] == 1)].iloc[:, :4].mean()
versicolor_std = df[(df['target'] == 1)].iloc[:, :4].std()

virginica_mean = df[(df['target'] == 2)].iloc[:, :4].mean()
virginica_std = df[(df['target'] == 2)].iloc[:, :4].std()

likelihoods = []

for feature in list(df.iloc[:,:4]):
    likelihood = []
    for target in range(len(targets)):
        mean = df[(df['target'] == target)].iloc[:, :4].mean()
        std = df[(df['target'] == target)].iloc[:, :4].std()
        likelihood.append(normalizator(mean, std, feature, test_data))
    likelihoods.append(likelihood)

for lh, feature in zip(likelihoods, list(df.iloc[:,:4])):
    print(feature)
    for i, target in enumerate(targets):
        print(target, ': ', lh[i])


print(likelihoods)

setosa_posterial = 1/3
versicolor_posterial = 1/3
virginica_posterial = 1
for idx in range(len(likelihoods)):
    setosa_posterial = likelihoods[idx][0] * setosa_posterial
    versicolor_posterial = likelihoods[idx][1] * versicolor_posterial
    virginica_posterial = likelihoods[idx][2] * virginica_posterial

print(setosa_posterial)
print(versicolor_posterial)
print(virginica_posterial)


