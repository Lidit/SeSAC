import numpy as np

scalar_np = np.array(3.14)
vec_np = np.array([1, 2, 3])
mat_np = np.array([[1, 2], [3, 4]])
tensor_np = np.array([[[1, 2, 3],
                       [4, 5, 6]],

                      [[11, 12, 13],
                       [14, 15, 16]]])

print(scalar_np.shape)
print(vec_np.shape)
print(mat_np.shape)
print(tensor_np.shape)
