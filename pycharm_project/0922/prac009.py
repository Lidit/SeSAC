# matrix Addition

mat1 = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

mat2 = [[11, 12, 13],
        [14, 15, 16],
        [17, 18, 19]]

mat3 = list()

for dim1_idx in range(len(mat1)):
    vec = list()
    for dim2_idx in range(len(mat1[0])):
        vec.append(mat1[dim1_idx][dim2_idx]+mat2[dim1_idx][dim2_idx])
    mat3.append(vec)

print('multiplication of mat1 and mat2', mat3)

