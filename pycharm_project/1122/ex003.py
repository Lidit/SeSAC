import numpy as np
import matplotlib.pyplot as plt

white_patch = 255 * np.ones(shape=(10, 10))
black_patch = 0 * np.ones(shape=(10, 10))

img1 = np.hstack([white_patch, black_patch, white_patch])
img2 = np.hstack([black_patch, white_patch, black_patch])

img = np.vstack([img1, img2, img1])

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
