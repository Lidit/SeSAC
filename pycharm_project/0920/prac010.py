# Euclidean distance(3)

v1, v2 = [1, 2, 3], [3, 4, 5]

diff_square_sum = 0

for dim_dix in range(len(v1)):
    diff_square_sum += (v1[dim_dix] - v2[dim_dix]) ** 2

e_distance = diff_square_sum ** 0.5

print("Euclidean Distance between v1 and v2: ", e_distance)
