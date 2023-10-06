import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10,5))

uniform = np.random.rand(1000)
ax.hist(uniform)
print(uniform.shape)
print(uniform.mean())
print(uniform.std())

plt.show()
