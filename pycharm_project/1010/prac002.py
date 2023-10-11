# code1_normal histogram
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)
x_data = np.random.normal(5, 5, (100,))

fig, ax = plt.subplots(figsize=(5, 5))

ax = plt.hist(x_data, rwidth=0.9)

plt.show()
