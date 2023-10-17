from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import tree

diabetes = load_diabetes()

X, y = load_diabetes(return_X_y=True)

data, targets = diabetes.data, diabetes.target

print(data.shape, targets.shape, '\n')
print(diabetes.feature_names)

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=7)

model = model.fit(X_train, y_train)

print("depth:", model.get_depth())
print("number of leaves:", model.get_n_leaves())

score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)

print("model score: ", score)
print("R^2 score: ", r2)

plt.figure(figsize=(30, 12))
tree.plot_tree(model,
               feature_names=diabetes.feature_names,
               filled=True,
               rounded=True)
plt.show()
