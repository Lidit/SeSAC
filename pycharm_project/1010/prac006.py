# targets
import numpy as np

for idx in range(4):
    print(np.full((100,), idx))

# n_classes = 4
# n_data = 100
# data = []
# for class_idx in range(n_classes):
#     data_ = class_idx * np.ones(100,)
#     data.append(data_)
# data = np.hstack(data)
# # data = np.concatenate(data)
# print(data.shape)
