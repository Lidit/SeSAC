# boxplot 5
import matplotlib.pyplot as plt
import numpy as np

n_student = 100
math_score = np.random.normal(loc=50, scale=10, size=(100,))

fig, ax = plt.subplots(figsize=(7, 7))

medianprops = {'linewidth': 2, 'color': 'k'}
boxprops = {'linestyle': '--', 'color': 'k', 'alpha': 0.7}
whiskerprops = {'linestyle': '--', 'color': 'tab:blue', 'alpha': 0.7}

ax.boxplot(math_score, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops)

plt.show()
