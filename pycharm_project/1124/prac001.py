import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
white_patch = 1 * np.ones(shape=(10, 10))
black_patch = 0 * np.ones(shape=(10, 10))

img1 = np.hstack([white_patch, black_patch])
img2 = np.hstack([black_patch, white_patch])

data = np.vstack([img1, img2])
data = np.tile(data, reps=[2, 2])

filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# H, W = data.shape
# F = filter_x.shape[0]
# H_ = H - F + 1
# W_ = W - F + 1

# filterd_data = np.zeros(shape=(H_,W_))

# for h_idx in range(H_):
#     for w_idx in range(W_):
#         window = data[h_idx:h_idx+F, w_idx:w_idx+F]
#         z1 = np.sum(window * filter_x)
#         # z2 = np.sum(window * filter_y)
#         # z = np.max((z1,z2))
#         filterd_data[h_idx,w_idx] = z1

# for h_idx in range(H_):
#     for w_idx in range(W_):
#         window = data[h_idx:h_idx+F, w_idx:w_idx+F]
#         z = np.sum(window * filter_y)
#         filterd_data[h_idx,w_idx] = z

# print(filterd_data)

#
# Ly, Lx, F_x, F_y = len(data), len(data), len(filter_x), len(filter_y)
Lx, Fx = len(data[0]), len(filter_x[0])
# Ly, Fy = len(data), len(filter_x)
Lx_ = Lx - Fx + 1
# Ly_ = Ly - Fy + 1
# L_x = Lx - F_x + 1
# L_y = Ly - F_y + 1

filter_x_idx = np.arange(Fx).reshape(1,-1)
window_x_idx = np.arange(Lx_).reshape(-1,1)
# filter_y_idx = np.arange(Fy).reshape(1,-1)
# window_y_idx = np.arange(Ly_).reshape(-1,1)
# print(filter_idx,'\n')
# print(window_idx)
idx_arr_x = filter_x_idx + window_x_idx
# idx_arr_y = filter_y_idx + window_y_idx
# print(idx_arr)

window_mat = data[idx_arr_x]
# print(window_mat[0,0])

print(window_mat.shape, filter_x.shape)
# idx_arr = filter_idx+window_idx
#
# window_mat = data[idx_arr,idx_arr]
#
# print(window_mat.shape, filter_x.shape)
#
collections = np.matmul(window_mat, filter_x.reshape())
print(collections)
#
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(collections, cmap='gray')

ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.tight_layout()
plt.show()
