# ndarray object of numpy
import numpy as np

a = np.array([1, 2, 3])
print(type(a))

for attr in dir(a):
    print(attr)
