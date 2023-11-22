import numpy as np
import matplotlib.pyplot as plt

img = np.arange(0,256,50).reshape(1,-1)
img = img.repeat(100,axis=0).repeat(30,axis=1)

fig, ax = plt.subplots(figsize=(8,4))
ax.imshow(img, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
fig.tight_layout()
plt.show()