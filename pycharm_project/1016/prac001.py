from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

data, targets = iris.data, iris.target

print(data.shape, targets.shape, '\n')
print(data)
print(targets)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=11)

# print(f"{type(X_train) = } / {X_train.shape = }")
# print(f"{type(X_test) = } / {X_test.shape = }")
# print(f"{type(y_train) = } / {y_train.shape = }")
# print(f"{type(y_test) = } / {y_test.shape = }\n")

model = DecisionTreeClassifier()

for attr in dir(model):
    if not attr.startswith("__"):
        print(attr)

model.fit(X_train, y_train)
print("depth:", model.get_depth())
print("number of leaves:", model.get_n_leaves())

accuracy = model.score(X_test, y_test)
# print(f"{accuracy = :.4f}")

import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20,15))
tree.plot_tree(model,
               class_names=iris.target_names,
               impurity=True, filled=True,
               rounded=True)

plt.show()
