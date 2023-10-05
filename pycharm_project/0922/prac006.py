# 과목별 최고점, 최우수 학생 구하기

scores = [[10, 40, 20],
          [50, 20, 60],
          [70, 40, 30],
          [30, 80, 40]]

M_class_score = list()
M_score_indices = list()

for class_idx in range(len(scores[0])):
    M_score = 0
    M_indices = 0
    for student_idx in range(len(scores)):
        if M_score == 0:
            M_score = scores[student_idx][class_idx]
            M_indices = class_idx
        elif M_score < scores[student_idx][class_idx]:
            M_score = scores[student_idx][class_idx]
            M_indices = student_idx
        else:
            pass
    M_class_score.append(M_score)
    M_score_indices.append(M_indices)

print("Max scores:", M_class_score)
print("Max score indices: ", M_score_indices)
