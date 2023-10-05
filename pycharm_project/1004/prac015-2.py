# ax.violinplot

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

fig, ax = plt.subplots(figsize=(7, 7))

data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(5, 2, 200)
data3 = np.random.normal(13, 3, 300)

xticks = np.arange(3)

# quantiles

ax.violinplot([data1, data2, data3], quantiles=[[0.25, 0.75], [0.1, 0.9], [0.3, 0.9]], positions=xticks)

ax.set_xticks(xticks)
ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])
ax.set_xlabel('Species', fontsize=15)
ax.set_ylabel('Values', fontsize=15)

plt.show()
