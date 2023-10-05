# Euclidean Distance(4)

vectors = [[1, 11],
           [2, 12],
           [3, 13],
           [4, 14]]

e_distance = 0

for dim1_idx in range(len(vectors)):
    diff = 0
    for dim2_idx in range(len(vectors[0])):
        if diff == 0:
            diff = vectors[dim1_idx][dim2_idx]
        else:
            diff -= vectors[dim1_idx][dim2_idx]
    e_distance += diff ** 2

e_distance **= 0.5

print("Euclidean distance: ", e_distance)
