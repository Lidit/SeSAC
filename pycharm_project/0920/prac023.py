# Basic usage of if statement with for-loop(4)

numbers = list()
for num in range(20):
    numbers.append(num)

numbers.append(3.14)
print(numbers)

for num in numbers:
    if num % 2 == 0:
        print("Even number")
    elif num % 2 == 1:
        print("Odd number")
    else:
        print("not an Integer")
