class Slime:
    def __init__(self):
        self.HP = 30
        self.damage = 5
        self.kill_exp = 50
        self.kill_money = 10

    def get_HP(self):
        return self.HP

    def set_HP(self, hp):
        self.HP = hp

    def get_kill_exp(self):
        return self.kill_exp

    def get_kill_money(self):
        return self.kill_money

class Wolf:
    def __init__(self):
        self.HP = 50
        self.damage = 10
        self.kill_exp = 80
        self.kill_money = 20

    def get_HP(self):
        return self.HP

    def set_HP(self, hp):
        self.HP = hp

    def get_kill_exp(self):
        return self.kill_exp

    def get_kill_money(self):
        return self.kill_money