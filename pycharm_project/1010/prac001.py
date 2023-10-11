# Euclidean distance(3)

d5, d12 = [2.75, 7.5] , [5, 2.5]

diff_square_sum = 0
m_distance = 0

for dim_dix in range(len(d5)):
    diff_square_sum += (d5[dim_dix] - d12[dim_dix]) ** 2
    m_distance += abs(d5[dim_dix] - d12[dim_dix])

e_distance = diff_square_sum ** 0.5


print("Euclidean Distance between d5 and d12: ", e_distance)
print("Mangattan Distance between d5 and d12: ", m_distance)

d17 = [5.25, 9.5]
diff_square_sum = 0
m_distance = 0

for dim_dix in range(len(d17)):
    diff_square_sum += (d12[dim_dix] - d17[dim_dix]) ** 2
    m_distance += abs(d12[dim_dix] - d17[dim_dix])

e_distance = diff_square_sum ** 0.5

print("Euclidean Distance between d12 and d17: ", e_distance)
print("Mangattan Distance between d12 and d17: ", m_distance)
