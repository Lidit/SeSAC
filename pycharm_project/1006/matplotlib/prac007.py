import matplotlib.pyplot as plt
import numpy as np

n_student = 100
math_score = np.random.normal(loc=50, scale=15, size=(100, 1))
chem_score = np.random.normal(loc=70, scale=10, size=(n_student, 1))
phy_score = np.random.normal(loc=30, scale=12, size=(n_student, 1))
pro_score = np.random.normal(loc=80, scale=5, size=(n_student, 1))
data = np.hstack((math_score, chem_score, phy_score, pro_score))

medianprops = {'linewidth': 1.5, 'color': 'red'}
boxprops = {'linewidth': 1.5, 'color': 'k', 'alpha': 0.7}
whiskerprops = {'linestyle': '--', 'color': 'tab:blue', 'alpha': 0.8}

labels = ['Math', 'English', 'Physics', 'Programming']
plt.style.use("seaborn")

xticks = np.arange(4)

fig, axes = plt.subplots(2,1,figsize=(15, 10))

axes[0].boxplot(data,medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
           capprops=whiskerprops)
axes[1].violinplot(data, positions=xticks)

axes[0].set_ylim([0, 100])
axes[1].set_ylim([0, 100])

axes[0].tick_params(labelsize=20, bottom=False,labelbottom=False)
axes[1].tick_params(labelsize=20)

axes[1].set_xticks(xticks)
axes[1].set_xticklabels(labels)

fig.subplots_adjust(hspace=0.1)

plt.show()
