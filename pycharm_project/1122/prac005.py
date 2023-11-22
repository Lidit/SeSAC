import numpy as np
import matplotlib.pyplot as plt

# img = np.arange(255, 0, -50).reshape(-1, 1)
img1 = np.arange(0, 256,1).reshape(1, -1)
img2 = np.arange(0, 256,1)[::-1].reshape(1, -1)

img1 = img1.repeat(200, axis=0).repeat(1, axis=1)
img2 = img2.repeat(200, axis=0).repeat(1, axis=1)

img = np.vstack([img1, img2])

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
