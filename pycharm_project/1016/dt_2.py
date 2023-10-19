from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


data = pd.read_csv('bike_sharing.csv')

# [instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt]
# print(data.head(1))

X_data = data[['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]
y_data = data['cnt']

# print(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)

model = DecisionTreeRegressor(max_depth=5)

model = model.fit(X_train, y_train)

score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("depth:", model.get_depth())
print("number of leaves:", model.get_n_leaves())

print("model score: ", score)
print("R^2 score: ", r2)

columns = np.array(data.columns)
# data[columns[0]]

plt.figure(figsize=(15, 10))
tree.plot_tree(model,
               feature_names=data.columns,
               filled=True,
               rounded=True)
plt.tight_layout()
plt.show()
