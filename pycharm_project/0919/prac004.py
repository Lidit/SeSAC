# Making Unit Vectors
# 단위벡터: 크기가 1인 벡터
# 벡터의 각 성분을 놈으로 나누어 준다
# 벡터를 유닛벡터로 변환하는 것을 표준화로 볼 수 있다

x, y, z = 1, 2, 3

norm = (x**2 + y**2 + z**2) ** 0.5
print(norm)

x, y, z = x/norm, y/norm, z/norm
norm = (x**2 + y**2 + z**2) ** 0.5
print(norm)
