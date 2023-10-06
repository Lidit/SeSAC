import numpy as np

a = np.arange(24)
b = np.reshape(a, (2, 3, 4))

print('original ndarray: \n', a)
print('reshaped ndarray: \n', b)
