# Mean subtraction

# 숫자 리스트 선언
scores = [10, 20, 30]
n_student = len(scores)

mean = (scores[0] + scores[1] + scores[2])/n_student
print("score mean: ", mean)

# 각 원소를 평균으로 빼기
scores[0] -= mean
scores[1] -= mean
scores[2] -= mean

mean = (scores[0] + scores[1] + scores[2])/n_student
print("score mean: ", mean)
