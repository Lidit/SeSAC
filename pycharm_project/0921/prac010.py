# mean subtraction

scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]
n_class = len(scores[0])
n_student = len(scores)
class_score_means = list()
class_score_sum = list()

for _ in range(n_class):
    class_score_sum.append(0)

for student_idx in range(n_student):
    for class_idx in range(n_class):
        class_score_sum[class_idx] += scores[student_idx][class_idx]

for sum_class in class_score_sum:
    class_score_means.append(sum_class / n_student)

print("---before mean subtraction---")
print(scores)
print("sum of classes's score: ", class_score_sum)
print("mean of classes's scores: ", class_score_means)

# Mean subtraction
for student_idx in range(n_student):
    for class_idx in range(n_class):
        scores[student_idx][class_idx] -= class_score_means[class_idx]

# Mean subtraction 이후 평균 구하기
class_score_means = list()
class_score_sum = list()
for _ in range(n_class):
    class_score_sum.append(0)

for student_idx in range(n_student):
    for class_idx in range(n_class):
        class_score_sum[class_idx] += scores[student_idx][class_idx]

for sum_class in class_score_sum:
    class_score_means.append(sum_class / n_student)

print("---after mean subtraction---")
print(scores)
print("sum of classes's score: ", class_score_sum)
print("mean of classes's scores: ", class_score_means)
