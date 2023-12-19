import os
import locale
import platform


# 로거 이름
LOGGER_NAME = 'rl_stock'


# 경로 설정
BASE_DIR = os.environ.get('RLTRADER_BASE',
    os.path.abspath(os.path.join('project2_rl')))


# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
