import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')

fig, ax = plt.subplots(figsize=(10,5))

uniform = np.random.uniform(low=-10, high=10, size=(10000,))
ax.hist(uniform, bins=20)
print(uniform.shape)
print(uniform.mean())
print(uniform.std())

plt.show()
