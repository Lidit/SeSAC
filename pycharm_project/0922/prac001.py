# Hadamard Product(5)

vectors = [[1, 11, 21],
           [2, 12, 22],
           [3, 13, 23],
           [4, 14, 24]]
v_hadamard = list()

for dim1_idx in range(len(vectors)):
    hadamard = 1
    for dim2_idx in range(len(vectors[0])):
        hadamard *= vectors[dim1_idx][dim2_idx]
    v_hadamard.append(hadamard)

print(v_hadamard)

