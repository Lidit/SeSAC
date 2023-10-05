# ax.axvline and ax.axhline

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4*np.pi, 4*np.pi, 200)
sin = np.sin(x)

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(x,sin)

ax.axhline(y=1, ls=':', lw=1, color='grey')
ax.axhline(y=-1, ls=':', lw=1, color='grey')

plt.show()
