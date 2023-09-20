# get min-max value & index in list using for-loop and if statement

scores = [60, -20, 40, 120, 70]
M, m = None, None
M_idx, m_idx = 0, 0

for score_idx in range(len(scores)):
    score = scores[score_idx]

    if M is None or score > M:
        M = score
        M_idx = score_idx
    if m is None or score < m:
        m = score
        m_idx = score_idx

print("M/M_idx: ", M, M_idx)
print("m/m_idx: ", m, m_idx)
