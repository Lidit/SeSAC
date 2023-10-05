import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7, 7), facecolor='linen')

ax = fig.add_subplot()
ax.plot([2, 3, 1])
ax.scatter([2, 3, 1], [2, 3, 4])
plt.show()
