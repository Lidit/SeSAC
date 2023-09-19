# Mean Squared Error(MSE)
# 평균제곱오차

pred1, pred2, pred3 = 10, 20, 30
y1, y2, y3 = 10, 25, 40
n_data = 3

s_error1 = (pred1 - y1)**2
s_error2 = (pred2 - y2)**2
s_error3 = (pred3 - y3)**2

mse = (s_error1 + s_error2 + s_error3)/n_data
print(mse)
