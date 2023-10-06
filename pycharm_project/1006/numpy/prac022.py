# Broadcasting in Numpy
import numpy as np

# when ndims are not equal

a = np.array(3)
u = np.arange(5)

print("shape: {}/{}".format(a.shape, u.shape))
print("a: ", a)
print("u: ", u, '\n')

print('a*u', a*u)
