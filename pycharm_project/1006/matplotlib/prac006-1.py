# boxplot 6-1
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

fig, ax = plt.subplots(figsize=(10, 7))
ax.boxplot(data, notch=True, showfliers=False, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
           capprops=whiskerprops)

ax.set_ylim([0, 100])

major_yticks = np.arange(0, 101, 20)
minor_yticks = np.arange(0, 101, 5)
ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor=True)

ax.tick_params(labelsize=20)
ax.tick_params(axis='x', rotation=10)

ax.grid(axis='y', linewidth=2)
ax.grid(axis='y', which='minor', linewidth=2, linestyle=':')
ax.grid(axis='x', linewidth=0)

plt.show()
