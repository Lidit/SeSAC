def f(x): return 1 / 10 * ((x - 2) ** 2)


def df_dx(x): return 1 / 5 * (x - 2)


x = 3
ITERATIONS = 50

print(f'Initial x:{x}')

for iter in range(ITERATIONS):
    dy_dx = df_dx(x)
    x = x - dy_dx
    print(f"{iter + 1} -th x: {x:.4f}")

x = -3
print(f'Initial x:{x}')
for iter in range(ITERATIONS):
    dy_dx = df_dx(x)
    x = x - dy_dx
    print(f"{iter + 1} -th x: {x:.4f}")
