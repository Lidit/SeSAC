import itertools
import pandas as pd
import numpy as np

# exchange_rates = {
#     # 환율 정보 입력
#     ('USD', 'EUR'): 0.6981,
#     ('USD', 'JPY'): 93.04,
#     ('USD', 'KRW'): 1166.1,
#     ('USD', 'MXN'): 13.0957,
#     ('USD', 'AUD'): 1.1145,
#     ('USD', 'CNY'): 6.8270,
#     ('USD', 'RUB'): 30.1000,
#     ('USD', 'CHF'): 1.0351,
#     ('EUR', 'JPY'): 133.26,
#     ('EUR', 'USD'): 1.4326,
#     ('EUR', 'KRW'): 1670.5,
#     ('EUR', 'MXN'): 18.1633,
#     ('EUR', 'AUD'): 1.5674,
#     ('EUR', 'CNY'): 9.4651,
#     ('EUR', 'RUB'): 43.427,
#     ('EUR', 'CHF'): 1.4825,
#     ('JPY', 'USD'): 0.010749,
#     ('JPY', 'EUR'): 0.00750,
#     ('JPY', 'KRW'): 12.5351,
#     ('JPY', 'MXN'): 0.1451,
#     ('JPY', 'AUD'): 1.2523,
#     ('JPY', 'CNY'): 0.07350,
#     ('JPY', 'RUB'): 0.3235,
#     ('JPY', 'CHF'): 0.0111,
#     ('KRW', 'USD'): 0.0009,
#     ('KRW', 'EUR'): 0.0007,
#     ('KRW', 'JPY'): 0.0798,
#     ('KRW', 'MXN'): 0.0112,
#     ('KRW', 'AUD'): 0.001,
#     ('KRW', 'CNY'): 0.0059,
#     ('KRW', 'RUB'): 0.0255,
#     ('KRW', 'CHF'): 0.000800,
#     ('MXN', 'USD'): 0.07640,
#     ('MXN', 'EUR'): 0.05330,
#     ('MXN', 'JPY'): 7.1035,
#     ('MXN', 'KRW'): 89.00,
#     ('MXN', 'AUD'): 0.08510,
#     ('MXN', 'CNY'): 0.5213,
#     ('MXN', 'RUB'): 2.3148,
#     ('MXN', 'CHF'): 0.07900,
#     ('AUD', 'USD'): 0.8972,
#     ('AUD', 'EUR'): 0.6263,
#     ('AUD', 'JPY'): 83.47,
#     ('AUD', 'KRW'): 1046.26,
#     ('AUD', 'MXN'): 11.7501,
#     ('AUD', 'CNY'): 6.1255,
#     ('AUD', 'RUB'): 27.20,
#     ('AUD', 'CHF'): 0.9285,
#     ('CNY', 'USD'): 0.1465,
#     ('CNY', 'EUR'): 0.1022,
#     ('CNY', 'JPY'): 13.6260,
#     ('CNY', 'KRW'): 170.81,
#     ('CNY', 'MXN'): 1.91820,
#     ('CNY', 'AUD'): 0.16330,
#     ('CNY', 'RUB'): 4.4088,
#     ('CNY', 'CHF'): 0.15160,
#     ('RUB', 'USD'): 0.03300,
#     ('RUB', 'EUR'): 0.023000,
#     ('RUB', 'JPY'): 3.0687,
#     ('RUB', 'KRW'): 38.470,
#     ('RUB', 'MXN'): 0.43200,
#     ('RUB', 'AUD'): 0.031700,
#     ('RUB', 'CNY'): 0.22530,
#     ('RUB', 'CHF'): 0.034100,
#     ('CHF', 'USD'): 0.96640,
#     ('CHF', 'EUR'): 0.67460,
#     ('CHF', 'JPY'): 89.900,
#     ('CHF', 'KRW'): 1126.9,
#     ('CHF', 'MXN'): 12.6554,
#     ('CHF', 'AUD'): 1.07710,
#     ('CHF', 'CNY'): 6.5974,
#     ('CHF', 'RUB'): 29.300,
# }

df = pd.read_excel('exchange_rate_predictions_3.xlsx', index_col=0)
df.columns = df.columns.str.replace('=X','')
# print(df.head())
# print(df.iloc[0].values())
# # for i in df[:]:
# #     print(i)

# 데이터프레임 생성 예시
# data = {'AB=X': [1, 2, 3, 4, 5], 'CD=Y': [6, 7, 8, 9, 10]}
# df = pd.DataFrame(data)
# 컬럼 이름을 두 글자 튜플로 변경
df.columns = [tuple(col[i:i+3] for i in range(0, len(col), 3)) for col in df.columns]
# 변경된 데이터프레임 출력
print(type(df.columns))
tp = tuple(df.columns)
print(tp[0])
exchange_rates ={}

for i in tp:
    exchange_rates[i] = df[i].values[0]

print(len(exchange_rates))





def generate_all_exchange_paths(exchange_rates, currencies):
    paths = []

    def dfs(path):
        nonlocal paths
        current_currency = path[-1]

        if len(path) == len(currencies):
            # 모든 나라를 포함하는 경로가 완성됨
            paths.append(path)
            return

        for next_currency in currencies:
            if next_currency not in path:
                if (current_currency, next_currency) in exchange_rates:
                    # 환율 정보가 있는 경우에만 다음 나라로 진행
                    dfs(path + [next_currency])

    for currency in currencies:
        dfs([currency])

    return paths

def find_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000):
    currencies = set(itertools.chain(*exchange_rates.keys()))

    optimal_path = None
    max_amount = start_amount

    all_paths = generate_all_exchange_paths(exchange_rates, currencies)

    for path in all_paths:
        if path[0] != start_currency or path[-1] != target_currency:
            continue

        amount = calculate_max_amount(exchange_rates, path, start_amount)

        if amount > max_amount:
            max_amount = amount
            optimal_path = path

    return optimal_path, max_amount

def calculate_max_amount(exchange_rates, path, start_amount):
    amount = start_amount
    for i in range(len(path) - 1):
        amount *= exchange_rates[(path[i], path[i + 1])]
    return amount

start_currency = 'USD'
target_currency = 'JPY'

optimal_path, max_amount = find_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000)

if optimal_path:
    print("Optimal Exchange Path:", " -> ".join(optimal_path))
    print("Maximum Amount:", max_amount)
else:
    print(f"No optimal arbitrage opportunities found for {start_currency} -> {target_currency}.")
