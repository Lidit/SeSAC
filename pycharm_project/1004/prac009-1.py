# ax.axvline and ax.axhline

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

# ax.axhline(y=1, color='black', linewidth=1)
ax.axhline(y=1, xmax=0.8, xmin=0.2, color='black', lw=1)

plt.show()
