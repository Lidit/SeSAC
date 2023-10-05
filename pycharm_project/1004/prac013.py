# bbox_to_anchor Argument
import matplotlib.pyplot as plt
import numpy as np

n_data = 100
random_noise1 = np.random.normal(0, 1, (n_data,))
random_noise2 = np.random.normal(1, 1, (n_data,))
random_noise3 = np.random.normal(2, 1, (n_data,))

fig, ax = plt.subplots(figsize=(15, 10))

ax.plot(random_noise1, label='random noise1')
ax.plot(random_noise2, label='random noise2')
ax.plot(random_noise3, label='random noise3')

ax.tick_params(labelsize=20)

ax.legend(fontsize=15, bbox_to_anchor=(2, 0), loc='center right')
fig.tight_layout()

plt.show()
