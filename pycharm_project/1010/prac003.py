# code2 dataset 1cluster
import numpy as np

# my code
# data = np.random.normal((5, 3), 1, (100, 2))
#
# print(np.mean(data, axis=0))
# print(np.std(data, axis=0))

# method 1
n_data = 100
x_data = np.random.normal(5, 1, size=(n_data,))
y_data = np.random.normal(3, 1, size=(n_data,))
print(x_data.shape)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)

data_ = np.hstack([x_data, y_data])

print(data_.shape)

# method2
n_data = 100
data = np.random.normal(loc=[5, 3], scale=[1, 1], size=(n_data, 2))

print(np.mean(data, axis=0))
print(np.std(data, axis=0))
