import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

# np.random.seed(1)

fig, ax = plt.subplots(figsize=(10, 5))

random_values = np.random.randn(300)
ax.hist(random_values, bins=20)
print(random_values.shape)

plt.show()
