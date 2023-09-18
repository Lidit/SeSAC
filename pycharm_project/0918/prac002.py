# mean subtraction

score1 = 10
score2 = 20
score3 = 30
n_student = 3

# 먼저 평균 구함
score_mean = (score1+score2+score3) / n_student
print(score_mean)

# 각 값에서 평균을 빼자
score1 -= score_mean
score2 -= score_mean
score3 -= score_mean

# 각 값-평균의 평균값 (mean subtraction)
mean = (score1 + score2 + score3)/n_student

print(mean)
