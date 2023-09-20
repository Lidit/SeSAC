# get min-max value in list using for-loop and if statement

scores = [60, 40, 70, 20, 30]
# not recommended. cuz you already know about range of data
M, m = 0, 100

for score in scores:
    if score > M:
        M = score
    if score < m:
        m = score

print("Max value: ", M)
print("min value: ", m)
