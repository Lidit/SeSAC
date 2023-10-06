# boxplot 4
import matplotlib.pyplot as plt
import numpy as np

n_student = 100
math_score = np.random.normal(loc=50, scale=10, size=(100,))

fig, ax = plt.subplots(figsize=(7, 7))

medianprops = {'linewidth': 2, 'color': 'k'}

ax.boxplot(math_score, medianprops=medianprops)

plt.show()
