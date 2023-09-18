import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 10))

ax = fig.add_subplot()

# ax.tick_params(labelsize=20, lenth=10, width=3, bottom=False, labelbottom=False, top=True, labeltop=True )
ax.tick_params(labelsize=20,
               lenth=10,
               width=3,
               bottom=False, labelbottom=False,
               left=False, labelleft=False,
               top=True, labeltop=True,
               right=True, labelright=True)

fig.tight_layout()

plt.show()