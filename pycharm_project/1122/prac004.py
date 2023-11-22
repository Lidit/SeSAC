import numpy as np
import matplotlib.pyplot as plt

# img = np.arange(255, 0, -50).reshape(-1, 1)
img = np.arange(0, 256, 50)[::-1].reshape(-1, 1)
img = img.repeat(30, axis=0).repeat(100, axis=1)

fig, ax = plt.subplots(figsize=(2, 4))
ax.imshow(img, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
