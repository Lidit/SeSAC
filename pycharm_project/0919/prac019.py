# Dot Product(2)
# Dot Product with list

v1, v2 = [1, 2, 3], [3, 4, 5]

# method1
dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
print(dot_product)

# method2
dot_product = 0
dot_product += v1[0]*v2[0]
dot_product += v1[1]*v2[1]
dot_product += v1[2]*v2[2]
print(dot_product)
