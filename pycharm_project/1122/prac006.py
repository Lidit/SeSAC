import numpy as np

data = 10 * np.arange(1, 11)
window_size = 3

print(data)

# for idx in range(0, len(data) - window_size + 1):
#     print(data[idx], data[idx + 1], data[idx + 2])

for i in range(len(data)-window_size+1):
    print(data[i:i+window_size])
