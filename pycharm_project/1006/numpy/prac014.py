# From random distributions

import numpy as np

normal = np.random.normal(loc=[-2, 0, 3], scale=[1, 2, 5], size=(200, 3))

print(normal)
print(normal.shape)
