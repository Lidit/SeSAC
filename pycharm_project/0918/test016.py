import matplotlib.pyplot as plt
import numpy as np

# ax.plot and ax.scatter

np.random.seed(0)

n_data = 100
x_data = np.random.normal(0, 1, (n_data,))
y_data = np.random.normal(0, 1, (n_data,))

fig, ax = plt.subplots(figsize=(7, 7))

# 둘다 동일한 scatter 를 출력해준다
ax.scatter(x_data, y_data)
# ax.plot(x_data, y_data, 'o') #plot을 scatter처럼

plt.show()

# outlier
