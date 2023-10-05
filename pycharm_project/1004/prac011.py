# basic usage of Legend
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

n_data = 100
random_noise1 = np.random.normal(0, 1, (n_data,))
random_noise2 = np.random.normal(1, 1, (n_data,))
random_noise3 = np.random.normal(2, 1, (n_data,))

fig, ax = plt.subplots(figsize=(10, 7))
ax.tick_params(labelsize=20)

ax.plot(random_noise1, label='random noise1')
ax.plot(random_noise2, label='random noise2')
ax.plot(random_noise3, label='random noise3')

# ax.legend()
ax.legend(fontsize=20)

plt.show()
