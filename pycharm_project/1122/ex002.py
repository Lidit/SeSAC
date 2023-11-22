import numpy as np
import matplotlib.pyplot as plt

tmp = np.ones(shape=(2, 3))  # (2,3) 의 shape을 가지고, 원소가 모두 1인 행렬 생성
print(tmp)

tmp2 = 10 * tmp  # 모든 원소에 scalar 곱
print(tmp2)
