import matplotlib.pyplot as plt
import numpy as np

figsize = (7, 7)

fig, ax = plt.subplots(figsize=figsize)
ax.set_title("title of a Ax")
ax.set_title("title of a Ax", fontsize=30)
ax.set_title("title of a Ax", fontsize=30, fontfamily='monospace')

plt.show()