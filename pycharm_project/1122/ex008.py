import numpy as np
import matplotlib.pyplot as plt

img = np.arange(0, 151, 50).reshape(1, -1)
img = img.repeat(120, axis=0).repeat(30, axis=1)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap='gray', vmax=255, vmin=0)

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
