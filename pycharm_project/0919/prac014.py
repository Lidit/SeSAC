# Hadamard Product with List

# method 1
v1, v2 = [1, 2, 3], [3, 4, 5]
v3 = [v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]]
print(v3)

# method 2
v1, v2 = [1, 2, 3], [3, 4, 5]
v3 = [0, 0, 0]

v3[0] = v1[0] * v2[0]
v3[1] = v1[1] * v2[1]
v3[2] = v1[2] * v2[2]
print(v3)
