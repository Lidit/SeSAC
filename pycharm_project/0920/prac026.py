# get min-max value in list using for-loop and if statement(2)

scores = [-20, 60, 40, 70, 120]

# method 1
M, m = scores[0], scores[0]

for score in scores:
    if score > M:
        M = score
    if score < m:
        n = score
print('Max value: ', M)
print('min value: ', m)

# method 2

M, m = None, None

for score in scores:
    if M is None or score > M:
        M = score
    if m is None or score < m:
        m = score

print('Max value: ', M)
print('min value: ', m)
