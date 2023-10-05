# Dot Product(4)

vectors = [[1, 11],
           [2, 12],
           [3, 13],
           [4, 14]]

dot_producted = 0

for dim1_idx in range(len(vectors)):
    dp = 1
    for dim2_idx in range(len(vectors[0])):
        dp *= vectors[dim1_idx][dim2_idx]
    dot_producted += dp

print(dot_producted)