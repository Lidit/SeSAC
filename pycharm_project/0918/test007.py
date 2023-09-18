import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(10, 10))

ax1 = fig.add_subplot()
ax2 = ax1.twinx()

ax1.set_xlim([0, 100])
ax1.set_ylim([0, 100])
ax2.set_ylim([0, 0.1])
ax1.set_title("Twinx Graph", fontsize=30)
ax1.set_ylabel('Data1', fontsize=20)
ax2.set_ylabel('Data1', fontsize=20)

fig.tight_layout()

plt.show()