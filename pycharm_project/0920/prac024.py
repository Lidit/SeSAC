# Basic usage of if statement with for-loop(5)

muliple_of = 3

numbers = list()

for num in range(100):
    numbers.append(num)

sum_multiple_of_n = 0

for num in numbers:
    if num % muliple_of == 0:
        sum_multiple_of_n += num

print(sum_multiple_of_n)
