# Dimensional Window Extraction
import numpy as np

data1 = 10 * np.arange(1, 8)
data2 = 10 * np.arange(0, 5).reshape(-1, 1)

data = data1 + data2
h, w = data.shape
window_size = 3

h_ = h - window_size + 1
l_ = w - window_size + 1

count = 0
for h_idx in range(h_):
    for l_idx in range(l_):
        count += 1
        print(data[h_idx:h_idx + window_size, l_idx:l_idx + window_size])


# print(count)