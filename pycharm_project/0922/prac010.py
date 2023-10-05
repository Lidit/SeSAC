# Matrix-vector Multiplication

mat = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]

vec = [10, 20, 30]

multiplied_vec = list()

for dim1_idx in range(len(mat)):
    dot_product = 0
    for dim2_idx in range(len(mat[0])):
        dot_product += vec[dim2_idx] * mat[dim1_idx][dim2_idx]
    multiplied_vec.append(dot_product)

print('multiplication of mat and vec: ', multiplied_vec)
