u = [1, 2, 3]
v = [4, 5, 6]

w = u + v
print(w)

w = []
for data_idx in range(len(u)):
    w.append(u[data_idx] + v[data_idx])

print(w)
#
# u_np = np.array(u)
# v_np = np.array(v)
#
# w_np = np.add(u_np, v_np)
#
# print(w_np)
