class Character:
    def __init__(self):
        self.lv = 1
        self.exp = 0
        self.HP = 100
        self.max_HP = 100
        self.damage = 10
        self.money = 100
        self.potion = 0

    def print_states(self):
        print("현재 레벨: ", self.lv)
        print("현재 경험치: ", self.exp)
        print("다음 레벨을 위한 경험치: ", (self.lv * 100) - self.exp)
        print("HP: ", self.HP)
        print("HP 최대치: ", self.max_HP)
        print("공격력: ", self.damage)
        print("돈: ", self.money)
        print('소지한 포션: ', self.potion)

    def save_states(self):
        f = open("save_file.csv", 'w')
        f.write(str(self.lv) + '\n')
        f.write(str(self.exp) + '\n')
        f.write(str(self.HP) + '\n')
        f.write(str(self.max_HP) + '\n')
        f.write(str(self.damage) + '\n')
        f.write(str(self.money) + '\n')
        f.write(str(self.potion) + '\n')
        f.close()

    def load_states(self):
        f = open("save_file.csv", 'r')
        lines = f.readlines()
        self.lv = int(lines[0].strip())
        self.exp = int(lines[1].strip())
        self.HP = int(lines[2].strip())
        self.max_HP = int(lines[3].strip())
        self.damage = int(lines[4].strip())
        self.money = int(lines[5].strip())
        self.potion = int(lines[6].strip())
        f.close()

    def check_level_up(self):
        if self.exp >= self.lv * 100: # 레벨업 조건이 만족될 때
            self.lv += 1 # 레벨을 1 올리고
            print(f"LEVEL UP!!! -> Lv.{self.lv}")

            self.max_HP += 10 # HP 최대치 10 증가
            self.set_HP(self.max_HP) # 레벨업 하면 HP 모두 채워주기
            self.damage += 3 # 데미지 3 추가

    def get_HP(self):
        return self.HP

    def set_HP(self, hp):
        self.HP = hp

    def attack_monster(self, monster):
        new_monster_HP = monster.get_HP() - self.damage
        monster.set_HP(new_monster_HP)

        if monster.get_HP() <= 0:
            kill_exp = monster.get_kill_exp()
            kill_money = monster.get_kill_money()

            self.exp += kill_exp
            self.money += kill_money
            print(f"\n경험치 {kill_exp}, 돈 {kill_money} 획득!")
            self.check_level_up()
            return None

        new_char_HP = self.get_HP() - monster.damage
        self.set_HP(new_char_HP)

    def drink_potion(self):
        if self.potion >= 1:
            if self.HP + 50 > self.max_HP:
                self.set_HP(self.max_HP)
            else:
                self.HP += 50
            self.potion -= 1
        else:
            print('마실 수 있는 포션이 없습니다.')

    def buy_potion(self):
        if self.money >= 30:
            self.money -= 30
            self.potion += 1
            print('포션을 1개 구매 했습니다.')
        else:
            print("소지금이 부족합니다.")

    # def check_level_up(self):
    #

