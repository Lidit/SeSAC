import matplotlib.pyplot as plt
import numpy as np

figsize= (7, 7)

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Title of figure')
fig.suptitle('Title of figure', fontsize=30)
fig.suptitle('Title of figure', fontsize=30, fontfamily='monospace')

plt.show()