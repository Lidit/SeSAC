import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
n_data = 500

data = np.random.normal(0,1,(n_data,))
fig, ax = plt.subplots(figsize=(14,10))

ax.tick_params(labelsize=30, length=10,width=3)
ax.hist(data, rwidth=0.9)

plt.show()
