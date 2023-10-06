# From random distributions
import numpy as np

normal = np.random.normal(loc=-2, scale=1, size=(3, 3))

print(normal)
print(normal.shape)
print(normal.mean())
