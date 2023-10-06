import numpy as np

a = np.random.randint(0,10, size=(2,2))

print(a)

row_vector = a.reshape(1,-1)
col_vector = a.reshape(-1, 1)

print(row_vector.shape, col_vector.shape)