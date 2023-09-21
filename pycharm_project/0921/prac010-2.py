# 정규화~

scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

n_class = len(scores[0])
class_score_means = list()

n_student = len(scores)
class_score_sum = list()
# 과목별 제곱
class_score_square_sum = list()
sqaure_scores = list()

for _ in range(n_student):
    student_scores = list()
    for _ in range(n_class):
        student_scores.append(0)
    sqaure_scores.append(student_scores)

for _ in range(n_class):
    class_score_sum.append(0)
    class_score_square_sum.append(0)

for student_idx in range(n_student):
    for class_idx in range(n_class):
        class_score_sum[class_idx] += scores[student_idx][class_idx]
        class_score_square_sum[class_idx] += scores[student_idx][class_idx] ** 2

# 평균
for sum_class in class_score_sum:
    class_score_means.append(sum_class / n_student)

# 과목별 제곱의 평균
mean_of_squares_of_class = list()
for square_sum in class_score_square_sum:
    mean_of_squares_of_class.append(square_sum / n_student)

# 과목별 평균의 제곱
square_of_mean_of_class = list()
for mean in class_score_means:
    square_of_mean_of_class.append(mean ** 2)

variances = list()
for variance_idx in range(len(square_of_mean_of_class)):
    variances.append(mean_of_squares_of_class[variance_idx] - square_of_mean_of_class[variance_idx])

stds = list()
for std_idx in range(len(variances)):
    stds.append(variances[std_idx] ** 0.5)
#
# print("sum of classes's score: ", class_score_sum)
# print("mean of classes's scores: ", class_score_means)
# print("mean of squares of classes: ", mean_of_squares_of_class)
# print("means of classes: ", square_of_mean_of_class)
# print("class score variance: ", variances)
# print("classes scores stds:", stds)

# 정규화 시작
for student_idx in range(n_student):
    for class_idx in range(n_class):
        scores[student_idx][class_idx] = (scores[student_idx][class_idx] - class_score_means[class_idx]) / stds[
            class_idx]

class_score_means = list()
class_score_sum = list()
# 과목별 제곱
class_score_square_sum = list()
sqaure_scores = list()

for _ in range(n_student):
    student_scores = list()
    for _ in range(n_class):
        student_scores.append(0)
    sqaure_scores.append(student_scores)

for _ in range(n_class):
    class_score_sum.append(0)
    class_score_square_sum.append(0)

for student_idx in range(n_student):
    for class_idx in range(n_class):
        class_score_sum[class_idx] += scores[student_idx][class_idx]
        class_score_square_sum[class_idx] += scores[student_idx][class_idx] ** 2

# 평균
for sum_class in class_score_sum:
    class_score_means.append(sum_class / n_student)

# 과목별 제곱의 평균
mean_of_squares_of_class = list()
for square_sum in class_score_square_sum:
    mean_of_squares_of_class.append(square_sum / n_student)

# 과목별 평균의 제곱
square_of_mean_of_class = list()
for mean in class_score_means:
    square_of_mean_of_class.append(mean ** 2)

variances = list()
for variance_idx in range(len(square_of_mean_of_class)):
    variances.append(mean_of_squares_of_class[variance_idx] - square_of_mean_of_class[variance_idx])

stds = list()
for std_idx in range(len(variances)):
    stds.append(variances[std_idx] ** 0.5)

print("---after standardization---", )
print(scores)
print("means of classes: ", square_of_mean_of_class)
print("classes scores stds:", stds)
