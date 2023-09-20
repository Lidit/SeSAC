# Vector Norm(3)
v1 = [1, 2, 3]

square_sum = 0
for dim_val in v1:
    square_sum += dim_val ** 2
norm = square_sum ** 0.5

print('Norm of v1:', norm)
