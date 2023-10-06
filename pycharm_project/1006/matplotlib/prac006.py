# boxplot 6
import matplotlib.pyplot as plt
import numpy as np

n_student = 100
math_score = np.random.normal(loc=50, scale=15, size=(100, 1))
chem_score = np.random.normal(loc=70, scale=10, size=(n_student, 1))
phy_score = np.random.normal(loc=30, scale=12, size=(n_student, 1))
pro_score = np.random.normal(loc=80, scale=5, size=(n_student, 1))

data = np.hstack((math_score, chem_score, phy_score, pro_score))

print(math_score.shape)
print(data.shape)

data_v = np.vstack((math_score, chem_score, phy_score, pro_score))
print(data_v.shape)
print(data)
print(data_v)

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylim([0, 100])

ax.boxplot(data)

plt.show()
