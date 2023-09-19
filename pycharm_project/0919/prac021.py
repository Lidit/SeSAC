# Mean Squared Error(2)
# 평균 제곱오차

predictions = [10, 20, 30]
labels = [10, 25, 40]
n_data = len(predictions)

mse = 0
mse += (predictions[0] - labels[0])**2
mse += (predictions[1] - labels[1])**2
mse += (predictions[2] - labels[2])**2
mse /= n_data
print(mse)
