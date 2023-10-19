from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

data = pd.read_csv('register_golf_club.csv')

print(data.head(1))

X_data = pd.get_dummies(data[['age', 'income', 'married', 'credit_score']])
y_data = pd.get_dummies(data['register_golf_club'])

# print(pd.get_dummies(y_data))
# y_data = pd.get_dummies(y_data)

encoded_y_data = pd.get_dummies(y_data)['yes']
print(X_data.head(0))
# encoded_y_data =
print(encoded_y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, encoded_y_data, test_size=0.2, random_state=12)

model = DecisionTreeClassifier(max_depth=1)

model = model.fit(X_train, y_train)

print("depth:", model.get_depth())
print("number of leaves:", model.get_n_leaves())

accuracy = model.score(X_test, y_test)
print("model score: ", accuracy)

plt.figure(figsize=(20,15))
tree.plot_tree(model,
               class_names=X_data.columns,
               impurity=True, filled=True,
               rounded=True)

plt.show()
