class LogicGate:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def __call__(self, x1, x2):
        if x1 * self.w1 + x2 * self.w2 + self.b <= 0:
            return 0
        else:
            return 1


class AndGate:
    def __init__(self):
        self.gate = LogicGate(0.5, 0.5, -0.7)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class NandGate:
    def __init__(self):
        self.gate = LogicGate(-0.5, -0.5, 0.7)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class OrGate:
    def __init__(self):
        self.gate = LogicGate(0.5, 0.5, -0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class NorGate:
    def __init__(self):
        self.gate = LogicGate(-0.5, -0.5, 0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)


class XorGate:
    def __init__(self):
        self.and_gate = AndGate()
        self.or_gate = OrGate()
        self.nand_gate = NandGate()

    def __call__(self, x1, x2):
        p = self.nand_gate(x1, x2)
        q = self.or_gate(x1, x2)
        return self.and_gate(p, q)


class XnorGate:
    def __init__(self):
        self.xor_gate = XorGate()

    def __call__(self, x1, x2):
        p = self.xor_gate(x1, x2)
        return self.xor_gate.nand_gate(p, p)

if __name__ == '__main__':


    xor_gate = XorGate()
    xnor_gate = XnorGate()

    print(xor_gate(0, 0))
    print(xor_gate(0, 1))
    print(xor_gate(1, 0))
    print(xor_gate(1, 1))

    print(xnor_gate(0, 0))
    print(xnor_gate(0, 1))
    print(xnor_gate(1, 0))
    print(xnor_gate(1, 1))

