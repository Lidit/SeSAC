from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


data = pd.read_csv('bike_sharing.csv', index_col=0)

# [instant, dteday, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt]
print(data.head(1))

# columns = np.array(data.columns)
# data[columns[0]]
