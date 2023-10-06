# boxplot 2
import matplotlib.pyplot as plt
import numpy as np

n_student = 100
math_score = np.random.normal(loc=50, scale=10, size=(100,))

fig, ax = plt.subplots(figsize=(7, 7))

# ax.boxplot(math_score, notch=True)
# ax.boxplot(math_score, notch=True, whis=2)
ax.boxplot(math_score, notch=True, whis=1, sym='bx')

plt.show()
