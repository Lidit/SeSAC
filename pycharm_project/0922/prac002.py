# Get vector norm

vectors = [[1, 11, 21],
           [2, 12, 22],
           [3, 13, 23],
           [4, 14, 24]]

norms = list()

for dim1_idx in range(len(vectors[0])):
    norm = 0
    for dim2_idx in range(len(vectors)):
        norm += vectors[dim2_idx][dim1_idx] ** 2
    norms.append(norm**0.5)

print(norms)
