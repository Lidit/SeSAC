# get row-wise mean with for-loop
# 학생별 평균점수 구하기

scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]
n_class = len(scores[0])
student_score_means = list()
n_student = len(scores)

for student_idx in range(n_student):
    student_score_sum = 0
    for class_idx in range(n_class):
        student_score_sum += scores[student_idx][class_idx]
    student_score_means.append(student_score_sum / n_class)

print("mean of student's scores: ", student_score_means)
