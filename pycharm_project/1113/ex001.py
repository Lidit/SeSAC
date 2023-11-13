def f(x): return 1 / 10 * x ** 2


def df_dx(x): return 1 / 5 * x


x = 3
ITERATIONS = 20

print(f'Initial x:{x}')

for iter in range(ITERATIONS):
    dy_dx = df_dx(x)
    x = x - dy_dx
    print(f"{iter + 1} -th x: {x:.4f}")
   