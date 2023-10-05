import matplotlib.pyplot as plt

figsize = (7, 7)

fig, ax = plt.subplots(figsize=figsize)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

ax.grid()
ax.tick_params(axis='both', labelsize=15)

ax.text(x=0, y=0, s='Hello1', fontsize=30)
ax.text(x=0.5, y=0, s='Hello2', fontsize=30)
ax.text(x=0.5, y=-0.5, s='Hello3', fontsize=30)

plt.show()
