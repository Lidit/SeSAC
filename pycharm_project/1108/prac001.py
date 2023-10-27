import os
from character import Character

from utils import game_main


if __name__ == '__main__':
    while True:
        next_move = int(input("\nSeSAC 온라인에 오신것을 환영 합니다.\n\n"
                              "1. 새로운 게임 시작하기\n"
                              "2. 지난 게임 불러오기\n"
                              "3. 게임 종료하기\n"
                              "다음 중 어떤 것을 하시겠습니까?"))

        if next_move == 1:
            print('\n새로운 캐릭터를 생성합니다.')
            char = Character()
            char.print_states()
            game_main(char)

        elif next_move == 2:
            if os.path.exists('save_file.csv'):
                print("\n저장된 파일을 불러옵니다.")
                char = Character()
                char.load_states()
                char.print_states()
                game_main(char)
            else:
                print("저장된 파일이 없습니다. 메인화면으로 돌아갑니다.")
                continue

        elif next_move == 3:
            print("\n게임을 종료합니다.")
            break
        else:
            print('\n잘못된 입력입니다. 메인화면으로 돌아갑니다.')
            continue
