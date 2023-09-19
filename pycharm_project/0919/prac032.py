# Mean subtraction(3)
# get Mean Subtraction using for-loop

scores = [10, 20, 30]

score_sum = 0
for score in scores:
    score_sum += score

score_mean = score_sum/len(scores)

# method1
scores_ms = list()

for score in scores:
    scores_ms.append(score - score_mean)

print(scores_ms)

# method2

for score_idx in range(len(scores)):
    scores[score_idx] -= score_mean
print(scores)
