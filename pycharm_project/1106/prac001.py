import numpy as np


def AND(x1, x2):
    if x2 > -x1 + 1.5:
        return 1
    else:
        return 0


def OR(x1, x2):
    if x2 >= -x1 + 0.5:
        return 1
    else:
        return 0


def NAND(x1, x2):
    if x2 <= -x1 + 1.2:
        return 1
    else:
        return 0


def XOR(x1, x2):
    g1 = NAND(x1, x2)
    g2 = OR(x1, x2)
    gate = AND(g1, g2)
    return gate


print("0 AND 0: ", AND(0, 0))
print("1 AND 0: ", AND(1, 0))
print("0 AND 1: ", AND(0, 1))
print("1 AND 1: ", AND(1, 1))

print("0 NAND 0: ", NAND(0, 0))
print("1 NAND 0: ", NAND(1, 0))
print("0 NAND 1: ", NAND(0, 1))
print("1 NAND 1: ", NAND(1, 1))
#
print("0 OR 0: ", OR(0, 0))
print("0 OR 1: ", OR(0, 1))
print("1 OR 0: ", OR(1, 0))
print("1 OR 1: ", OR(1, 1))

print("0 XOR 0: ", XOR(0, 0))
print("1 XOR 0: ", XOR(1, 0))
print("0 XOR 1: ", XOR(0, 1))
print("1 XOR 1: ", XOR(1, 1))
