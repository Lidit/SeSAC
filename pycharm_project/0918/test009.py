import matplotlib.pyplot as plt

figsize = (7, 7)

fig, ax = plt.subplots(figsize=figsize)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

ax.grid()
ax.tick_params(axis='both', labelsize=15)

# ax.text(x=0, y=0, s='Hello', fontsize=30)

plt.show()
