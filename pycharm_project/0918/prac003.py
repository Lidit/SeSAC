# 평균의 제곱과 제곱의 평균
score1 = 10
score2 = 20
score3 = 30
n_student = 3

# 평균
mean = (score1 + score2 + score3) / n_student

# 평균의 제곱
square_of_mean = mean ** 2

# 제곱의 평균
mean_of_square = (score1 ** 2 + score2 ** 2 + score3 ** 2) / n_student

print("square of mean: ", square_of_mean)
print("mean of square: ", mean_of_square)
