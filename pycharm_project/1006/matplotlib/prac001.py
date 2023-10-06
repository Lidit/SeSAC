import matplotlib.pyplot as plt
import numpy as np

# 1.5 IQR
# IQR == Q3 - Q1

n_student = 100
math_score = np.random.normal(loc=50, scale=10, size=(100,))

fig, ax = plt.subplots(figsize=(7, 7))
ax.boxplot(math_score)

plt.show()
