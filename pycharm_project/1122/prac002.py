import numpy as np
import matplotlib.pyplot as plt

white_patch = 255 * np.ones(shape=(10, 10))
gray_patch = 127.5 * np.ones(shape=(10, 10))
# white_patch = 1 * np.ones(shape=(10, 10))
# gray_patch = 0.5 * np.ones(shape=(10, 10))
black_patch = 0 * np.ones(shape=(10, 10))

img1 = np.hstack([white_patch, gray_patch])
img2 = np.hstack([gray_patch, black_patch])

img = np.vstack([img1, img2])

img = np.tile(img, reps=[4, 4])

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
