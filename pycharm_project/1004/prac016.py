import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

for attr in dir(iris):
    print(attr)

# # print(iris.data)
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.target)

feature_names = iris.feature_names
n_feature = len(feature_names)
species = iris.target_names
n_species = len(species)

iris_X, iris_y = iris.data, iris.target

print(iris_X)
print(iris_y)
print(feature_names)
print(species)
print(n_feature)