# transpose matrix

mat = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]

tp_mat = []

for dim1_idx in range(len(mat[0])):
    tmp_vec = list()
    for dim2_idx in range(len(mat)):
        tmp_vec.append(0)
    tp_mat.append(tmp_vec)

for dim1_idx in range(len(mat)):
    for dim2_idx in range(len(mat[0])):
        tp_mat[dim2_idx][dim1_idx] = mat[dim1_idx][dim2_idx]

print("transpose of mat", tp_mat)
