from prac001 import AND, NAND, OR, XOR


# half adder

def half_adder(x1, x2):
    s = XOR(x1, x2)
    c = AND(x1, x2)
    return s, c


# Full adder

def full_adder(x1, x2, ci):
    p = XOR(x1, x2)
    q = AND(p, ci)
    r = AND(x1, x2)
    s = XOR(p, ci)
    cout = OR(q, r)

    return s, cout


print("Full Adder x1: 0, x2: 0, Cin: 0 ", full_adder(0, 0, 0))
print("Full Adder x1: 0, x2: 1, Cin: 0 ", full_adder(0, 1, 0))
print("Full Adder x1: 1, x2: 0, Cin: 0 ", full_adder(1, 0, 0))
print("Full Adder x1: 1, x2: 1, Cin: 0 ", full_adder(1, 1, 0))
print("Full Adder x1: 0, x2: 0, Cin: 1 ", full_adder(0, 0, 1))
print("Full Adder x1: 0, x2: 1, Cin: 1 ", full_adder(0, 1, 1))
print("Full Adder x1: 1, x2: 0, Cin: 1 ", full_adder(1, 0, 1))
print("Full Adder x1: 1, x2: 1, Cin: 1 ", full_adder(1, 1, 1))


### 4bit adder

def adder_4bit(a, b):
    s0, c1 = half_adder(a[-1], b[-1])
    s1, c2 = full_adder(a[2], b[2], c1)
    s2, c3 = full_adder(a[1], b[1], c2)
    s3, c4 = full_adder(a[0], b[0], c3)

    return str(c4) + str(s3) + str(s2) + str(s1) + str(s0)


print("0010xb + 1111xb :", adder_4bit([0, 0, 1, 0], [1, 1, 1, 1])+"xb")
