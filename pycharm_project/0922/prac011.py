# Matrix-Matrix Multiplication

mat1 = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

mat2 = [[11, 12, 13],
        [14, 15, 16],
        [17, 18, 19]]

multiplied_mat = list()

for dim1_idx in range(len(mat1)):
    tmp_vec = []
    for dim2_idx in range(len(mat1[0])):
        tmp_vec.append(0)
    multiplied_mat.append(tmp_vec)

## 어려웡 ㅠㅠ

for mat1_idx in range(len(mat1)):
    for mat2_idx in range(len(mat1[0])):
        for multi_mat_idx in range(len(mat1)):
            multiplied_mat[mat1_idx][mat2_idx] += mat1[mat1_idx][multi_mat_idx] * mat2[multi_mat_idx][mat2_idx]

print("multiplication of mat1 and mat2", multiplied_mat)
