# Mean subtraction(2)

# 숫자 리스트 선언
scores = [10, 20, 30]
n_student = len(scores)

mean = (scores[0] + scores[1] + scores[2])/n_student
square_of_mean = mean**2 # 평균의 제곱
mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2)/n_student # 제곱의평균

variance = mean_of_square - square_of_mean # MOS - SOM
std = variance**0.5 # square root of variance

print("score mean: ", mean)

print("score standard deviation: ", std)
