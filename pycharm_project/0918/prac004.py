# 분산과 표준편차
score1 = 10
score2 = 20
score3 = 30
n_student = 3

score_mean = (score1 + score2 + score3) / n_student
square_of_mean = score_mean ** 2
mean_of_square = (score1 ** 2 + score2 ** 2 + score3 ** 2) / n_student

# variance: 분산 (제곱의평균 - 평균의 제곱)
score_variance = mean_of_square - square_of_mean

# standard deviation: 표준편차
score_std = score_variance ** 0.5

print("mean: ", score_mean)
print('variance: ', score_variance)
print('standard deviation: ', score_std)


def variance(**args):
    args_list = list(args)
    args_n = len(args)
    sum = 0
    for i in args:
        sum += i
