import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    gate = np.sum(w * x) + b
    if gate <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    gate = np.sum(w * x) + b
    if gate <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    gate = np.sum(w * x) + b
    if gate <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    g1 = NAND(x1, x2)
    g2 = OR(x1, x2)
    gate = AND(g1, g2)
    return gate


#
# def NOT(x):
#     if x == 0:
#         return 1
#     else:
#         return 0
#
#
# def XNOR(x1, x2):
#     return NOT(XOR(x1, x2))

# def non_linear(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([-1, 1])
#     b = 1
#     gate = np.sum(x * w) + b
#
#     if gate == 1:
#         return 1
#     else:
#         return 0


def non_linear1(x1, x2):
    if x2 >= x1 + 0.7:
        return 0
    else:
        return 1


def non_linear2(x1, x2):
    if x2 <= x1 - 0.7:
        return 0
    else:
        return 1


def non_linear3(x1, x2):
    return AND(non_linear1(x1, x2), non_linear2(x1, x2))


print("0 non-linear 0 =>", non_linear3(0, 0))
print("0 non-linear 1 =>", non_linear3(0, 1))
print("1 non-linear 0 =>", non_linear3(1, 0))
print("1 non-linear 1 =>", non_linear3(1, 1))
