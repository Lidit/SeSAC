# Making unit vectors(4)

vectors = [[1, 11, 21],
           [2, 12, 22],
           [3, 13, 23],
           [4, 14, 24]]

norms = list()
u_vectors = list()

for dim1_idx in range(len(vectors[0])):
    norm = 0
    for dim2_idx in range(len(vectors)):
        norm += vectors[dim2_idx][dim1_idx] ** 2
    norms.append(norm ** 0.5)

print(norms)

for dim1_idx in range(len(vectors[0])):
    for dim2_idx in range(len(vectors)):
        vectors[dim2_idx][dim1_idx] /= norms[dim1_idx]

for dim1_idx in range(len(vectors[0])):
    u_vec = 0
    for dim2_idx in range(len(vectors)):
        u_vec += vectors[dim2_idx][dim1_idx] ** 2
    u_vectors.append(u_vec ** 0.5)

print(u_vectors)
