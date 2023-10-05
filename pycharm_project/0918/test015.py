import matplotlib.pyplot as plt
import numpy as np

# color

color_list = [(1., 0., 0.),
              (0., 1., 0.),
              (0., 0., 1.)]

fig, ax = plt.subplots(figsize=(5, 10))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, len(color_list)])

for c_idx, c in enumerate(color_list):
    ax.text(0, c_idx, f"color={c}", fontsize=20, ha='center', color=c)

plt.show()
