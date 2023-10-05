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
violin = ax.violinplot([data1, data2, data3], showmeans=True, positions=xticks)

ax.set_xticks(xticks)
ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])
ax.set_xlabel('Species', fontsize=15)
ax.set_ylabel('Values', fontsize=15)

violin['bodies'][0].set_facecolor('blue')
violin['bodies'][1].set_facecolor('red')
violin['bodies'][2].set_facecolor('green')

violin['cbars'].set_edgecolor('red')
violin['cmaxes'].set_edgecolor('blue')
violin['cmins'].set_edgecolor('yellow')
violin['cmeans'].set_edgecolor('black')
plt.show()
