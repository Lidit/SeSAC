# making unit vector

v1 = [1,2,3]
square_sum = 0
for dim_val in v1:
    square_sum += dim_val ** 2

norm = square_sum ** 0.5

print('norm of v1: ', norm)

for dim_idx in range(len(v1)):
    v1[dim_idx] /= norm

# after make unit vector
square_sum = 0
for dim_val in v1:
    square_sum += dim_val ** 2
norm = square_sum ** 0.5

print("norm of v1: ", norm)
