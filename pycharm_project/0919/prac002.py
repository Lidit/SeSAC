# Scalar-Vector Operation
# numpy에선 broad-casting기능으로 각 차원이 다른 스칼라와 벡터간 연산을 할 수 있게 해준다
# 차원이 다른 변수간의 연산에선 shape이 중요?

# 스칼라
a = 10
# 벡터
x1, y1, z1 = 1, 2, 3

x2, y2, z2 = a * x1, a * y1, a * z1
x3, y3, z3 = a + x1, a + y1, a + z1
x4, y4, z4 = a - x1, a - y1, a - z1

print(x2, y2, z2)
print(x3, y3, z3)
print(x4, y4, z4)
