import matplotlib.pyplot as plt
import numpy as np

figsize = (7, 7)

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle("Title of a Figure", fontsize=30, color='darkblue', alpha=0.9)
ax.set_xlabel("X label", fontsize=20, color='darkblue', alpha=0.9)
ax.set_ylabel("Y label", fontsize=20, color='darkblue', alpha=0.9)

plt.show()
