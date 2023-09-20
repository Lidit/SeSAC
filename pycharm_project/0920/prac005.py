# Standardization(4)

math_scores, eng_scores = [50, 60, 70], [30, 40, 50]

n_student = len(math_scores)

math_sum, eng_sum = 0, 0
math_square_sum, eng_square_sum = 0, 0

for student_idx in range(n_student):
    math_sum += math_scores[student_idx]
    math_square_sum += math_scores[student_idx] ** 2

    eng_sum += eng_scores[student_idx]
    eng_square_sum += eng_scores[student_idx] ** 2

math_mean = math_sum/n_student
eng_mean = eng_sum/n_student

math_variance = math_square_sum/n_student - math_mean ** 2
eng_variance = eng_square_sum/n_student - eng_mean ** 2

math_std = math_variance ** 0.5
eng_std = eng_variance ** 0.5

# 정규화 시작
math_sum, eng_sum = 0, 0
math_square_sum, eng_square_sum = 0, 0

for student_idx in range(n_student):
    math_scores[student_idx] = (math_scores[student_idx] - math_mean) / math_std
    eng_scores[student_idx] = (eng_scores[student_idx] - eng_mean) / eng_std

for student_idx in range(n_student):
    math_sum += math_scores[student_idx]
    math_square_sum += math_scores[student_idx] ** 2

    eng_sum += eng_scores[student_idx]
    eng_square_sum += eng_scores[student_idx] ** 2

math_mean = math_sum / n_student
eng_mean = eng_sum / n_student

math_variance = math_square_sum / n_student - math_mean ** 2
eng_variance = eng_square_sum / n_student - eng_mean ** 2

math_std = math_variance ** 0.5
eng_std = eng_variance ** 0.5

print("Math Scores after standardization: ", math_scores)
print("English Scores after standardization: ", eng_scores)

print("mean/std of Math: ", math_mean, math_std)
print("mean/std of English: ", eng_mean, eng_std)

