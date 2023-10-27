from monsters import Slime, Wolf


def game_main(char):
    while True:
        action = input("\n\n1. 몬스터 잡기 2. 현재 상태 확인 3. 물약 사기(30원)\n"
                       "4. 게임 저장하기 5. 물약 마시기 0. 게임 종료\n"
                       "다음중 어떤 것을 하시겠습니까?")
        if action == '1':
            monster_idx = int(input("\n1. 슬라임\t"
                                    "2. 늑대\n"
                                    "어떤 몬스터를 잡을까요?"))
            if monster_idx == 1:
                new_battle(char, 'slime')
            elif monster_idx == 2:
                new_battle(char, 'wolf')
            else:
                print("잘못된 입력입니다. 메인으로 돌아갑니다.")

        elif action == '2':
            char.print_states()
        elif action == '3':
            char.buy_potion()
        elif action == '4':
            char.save_states()
        elif action == '5':
            char.drink_potion()
        elif action == '0':
            print("===== 게임 종료 =====\n")
            break


def new_battle(char, monster):
    if monster == 'slime':
        monster = Slime()
    elif monster == 'wolf':
        monster = Wolf()

    print("\nBattle 시작")
    print(f"내 HP: {char.get_HP()}, 적의 HP: {monster.get_HP()}")

    while True:
        action = int(input("1. 공격\t"
                           "2. 물약 먹기\t"
                           "3. 도망 가기\n"
                           "다음 중 어떤 것을 하시겠습니까? "))
        if action == 1:
            char.attack_monster(monster)
            if monster.get_HP() <= 0:
                print("적을 잡았습니다.")
                return None
            else:
                print("\n### 공격 후 ###")
                print(f"내 HP: {char.get_HP()}")
                print(f"적의 HP: {monster.get_HP()}\n")
        elif action == 2:
            char.drink_potion()
            print(f"남은 HP 포션: {char.potion}")
        if action == 3:
            print("도망갔습니다.")
            return None
    print("Battle 종료\n")
