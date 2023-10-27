# # import itertools
# #
# # # 환율 데이터 (가상 데이터)
# # exchange_rates = {
# #     ('USD', 'EUR'): 1.3863,
# #     ('EUR', 'JPY'): 0.79895,
# #     ('JPY', 'USD'): 90.313,
# #     ('USD', 'JPY'): 1.10725
# # }
# #
# # def simulate_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000):
# #     currencies = set(itertools.chain(*exchange_rates.keys()))
# #     optimal_path = None
# #     max_rate_of_return = -1  # 초기값 설정
# #
# #     for intermediate_currency in currencies:
# #         if (
# #             start_currency != intermediate_currency
# #             and intermediate_currency != target_currency
# #             and (start_currency, intermediate_currency) in exchange_rates
# #             and (intermediate_currency, target_currency) in exchange_rates
# #         ):
# #             start_to_intermediate_rate = exchange_rates[(start_currency, intermediate_currency)]
# #             intermediate_to_target_rate = exchange_rates[(intermediate_currency, target_currency)]
# #
# #             if start_to_intermediate_rate * intermediate_to_target_rate > 1:
# #                 rate_of_return = start_to_intermediate_rate * intermediate_to_target_rate - 1
# #
# #                 if rate_of_return > max_rate_of_return:
# #                     max_rate_of_return = rate_of_return
# #                     optimal_path = (start_currency, intermediate_currency, target_currency, rate_of_return)
# #
# #     if optimal_path:
# #         start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
# #         max_amount = start_amount
# #         # 최적 경로를 따라 환전
# #         max_amount *= exchange_rates[(start_currency, intermediate_currency)]
# #         max_amount *= exchange_rates[(intermediate_currency, end_currency)]
# #
# #         # 직접 환전한 경우의 수익률 계산
# #         direct_conversion_rate = exchange_rates.get((start_currency, target_currency), 0)
# #
# #         return max_amount, rate_of_return, direct_conversion_rate, optimal_path
# #     else:
# #         return start_amount, None
# #
# # start_currency = 'USD'
# # target_currency = 'JPY'
# #
# # max_amount, rate_of_return, direct_conversion_rate, optimal_path = simulate_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000)
# #
# # if optimal_path:
# #     start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
# #     print(f"Optimal Exchange Path: {start_currency} -> {intermediate_currency} -> {end_currency}")
# #     print(f"Maximum Exchange Amount: {max_amount:.4f} {end_currency}")
# #     print(f"Rate of Return (Compared to Direct Conversion): {rate_of_return - direct_conversion_rate:.4f}")
# # else:
# #     print(f"No optimal arbitrage opportunities found for {start_currency} -> {target_currency}.")
# import itertools
#
# # 환율 데이터 (가상 데이터)
# exchange_rates = {
#     ('USD', 'EUR'): 1.3863,
#     ('EUR', 'JPY'): 0.79895,
#     ('JPY', 'USD'): 90.313,
#     ('USD', 'JPY'): 1.10725
# }
#
# def simulate_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000):
#     currencies = set(itertools.chain(*exchange_rates.keys()))
#     optimal_path = None
#     max_rate_of_return = -1  # 초기값 설정
#
#     for intermediate_currency in currencies:
#         if (
#             start_currency != intermediate_currency
#             and intermediate_currency != target_currency
#             and (start_currency, intermediate_currency) in exchange_rates
#             and (intermediate_currency, target_currency) in exchange_rates
#         ):
#             start_to_intermediate_rate = exchange_rates[(start_currency, intermediate_currency)]
#             intermediate_to_target_rate = exchange_rates[(intermediate_currency, target_currency)]
#
#             if start_to_intermediate_rate * intermediate_to_target_rate > 1:
#                 rate_of_return = start_to_intermediate_rate * intermediate_to_target_rate - 1
#
#                 if rate_of_return > max_rate_of_return:
#                     max_rate_of_return = rate_of_return
#                     optimal_path = (start_currency, intermediate_currency, target_currency, rate_of_return)
#
#     if optimal_path:
#         start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
#         max_amount = start_amount
#         # 최적 경로를 따라 환전
#         max_amount *= exchange_rates[(start_currency, intermediate_currency)]
#         max_amount *= exchange_rates[(intermediate_currency, end_currency)]
#
#         # 직접 환전한 경우의 환율을 확인
#         direct_conversion_rate = exchange_rates.get((start_currency, target_currency), 0)
#
#         # 직접 환전한 경우의 화폐량 대비 수익률 계산
#         direct_conversion_return = (max_amount / start_amount) - 1
#
#         return max_amount, rate_of_return, direct_conversion_return, optimal_path
#     else:
#         return start_amount, None
#
# start_currency = 'USD'
# target_currency = 'JPY'
#
# max_amount, rate_of_return, direct_conversion_return, optimal_path = simulate_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000)
#
# if optimal_path:
#     start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
#     print(f"Optimal Exchange Path: {start_currency} -> {intermediate_currency} -> {end_currency}")
#     print(f"Maximum Exchange Amount: {max_amount:.4f} {end_currency}")
#     print(f"Rate of Return (Compared to Direct Conversion): {rate_of_return:.4f}")
#     print(f"Direct Conversion Rate of Return: {direct_conversion_return:.4f}")
# else:
#     print(f"No optimal arbitrage opportunities found for {start_currency} -> {target_currency}.")

import itertools

# 환율 데이터 (가상 데이터)
exchange_rates = {
    # 1 달러당 타깃 화폐액
    ('USD', 'EUR'): 0.6981,
    ('USD', 'JPY'): 93.04,
    ('USD', 'KRW'): 1166.1,
    ('USD', 'MXN'): 13.0957,
    ('USD', 'AUD'): 1.1145,
    ('USD', 'CNY'): 6.8270,
    ('USD', 'RUB'): 30.1000,
    ('USD', 'CHF'): 1.0351,
    # ('USD', 'INR'): 46.615,
    # 1 유로당 타깃 화폐액
    ('EUR', 'JPY'): 133.26,
    ('EUR', 'USD'): 1.4326,
    ('EUR', 'KRW'): 1670.5,
    ('EUR', 'MXN'): 18.1633,
    ('EUR', 'AUD'): 1.5674,
    ('EUR', 'CNY'): 9.4651,
    ('EUR', 'RUB'): 43.427,
    ('EUR', 'CHF'): 1.4825,
    # ('EUR', 'INR'): 66.7810,
    # 1 엔당 타깃 화폐액
    ('JPY', 'USD'): 0.010749,
    ('JPY', 'EUR'): 0.00750,
    ('JPY', 'KRW'): 12.5351,
    ('JPY', 'MXN'): 0.1451,
    ('JPY', 'AUD'): 1.2523,
    ('JPY', 'CNY'): 0.07350,
    ('JPY', 'RUB'): 0.3235,
    ('JPY', 'CHF'): 0.0111,
    # ('JPY', 'INR'): 0.0050,
    # 1₩ 당 타깃 화폐액
    ('KRW', 'USD'): 0.0009,
    ('KRW', 'EUR'): 0.0007,
    ('KRW', 'JPY'): 0.0798,
    ('KRW', 'MXN'): 0.0112,
    ('KRW', 'AUD'): 0.001,
    ('KRW', 'CNY'): 0.0059,
    ('KRW', 'RUB'): 0.0255,
    ('KRW', 'CHF'): 0.000800,
    # ('KRW', 'INR'): 0.0400,
    # 1 멕시코 페소당 타깃 화폐액
    ('MXN', 'USD'): 0.07640,
    ('MXN', 'EUR'): 0.05330,
    ('MXN', 'JPY'): 7.1035,
    ('MXN', 'KRW'): 89.00,
    ('MXN', 'AUD'): 0.08510,
    ('MXN', 'CNY'): 0.5213,
    ('MXN', 'RUB'): 2.3148,
    ('MXN', 'CHF'): 0.07900,
    # ('MXN', 'INR'): 3.5595,
    # 1 호주달러당 타깃 화폐액
    ('AUD', 'USD'): 0.8972,
    ('AUD', 'EUR'): 0.6263,
    ('AUD', 'JPY'): 83.47,
    ('AUD', 'KRW'): 1046.26,
    ('AUD', 'MXN'): 11.7501,
    ('AUD', 'CNY'): 6.1255,
    ('AUD', 'RUB'): 27.20,
    ('AUD', 'CHF'): 0.9285,
    # ('AUD', 'INR'): 41.825,
    # 1 중국 위안당 타깃 화폐액
    ('CNY', 'USD'): 0.1465,
    ('CNY', 'EUR'): 0.1022,
    ('CNY', 'JPY'): 13.6260,
    ('CNY', 'KRW'): 170.81,
    ('CNY', 'MXN'): 1.91820,
    ('CNY', 'AUD'): 0.16330,
    ('CNY', 'RUB'): 4.4088,
    ('CNY', 'CHF'): 0.15160,
    # ('CNY', 'INR'): 6.8281,
    # 1 러시아 루블당 타깃 화폐액
    ('RUB', 'USD'): 0.03300,
    ('RUB', 'EUR'): 0.023000,
    ('RUB', 'JPY'): 3.0687,
    ('RUB', 'KRW'): 38.470,
    ('RUB', 'MXN'): 0.43200,
    ('RUB', 'AUD'): 0.031700,
    ('RUB', 'CNY'): 0.22530,
    ('RUB', 'CHF'): 0.034100,
    # ('RUB', 'INR'): 1.53780,
    # 1 스위스 프랑당 타깃 화폐액
    ('CHF', 'USD'): 0.96640,
    ('CHF', 'EUR'): 0.67460,
    ('CHF', 'JPY'): 89.900,
    ('CHF', 'KRW'): 1126.9,
    ('CHF', 'MXN'): 12.6554,
    ('CHF', 'AUD'): 1.07710,
    ('CHF', 'CNY'): 6.5974,
    ('CHF', 'RUB'): 29.300,
    # ('CHF', 'INR'): 45.047,
    # 1 인도 루피당 타깃 화폐액
    # ('INR', 'USD'): 0.0215,
    # ('INR', 'EUR'): 0.012900,
    # ('INR', 'JPY'): 1.99560,
    # ('INR', 'KRW'): 25.015,
    # ('INR', 'MXN'): 0.28090,
    # ('INR', 'AUD'): 0.023900,
    # ('INR', 'CNY'): 0.15,
    # ('INR', 'RUB'): 0.64560,
    # ('INR', 'CHF'): 0.022200,
}


def simulate_optimal_exchange_path(exchange_rates, start_currency, target_currency, start_amount=10000):
    currencies = set(itertools.chain(*exchange_rates.keys()))
    optimal_path = None
    max_rate_of_return = -1  # 초기값 설정

    for intermediate_currency in currencies:
        if (
                start_currency != intermediate_currency
                and intermediate_currency != target_currency
                and (start_currency, intermediate_currency) in exchange_rates
                and (intermediate_currency, target_currency) in exchange_rates
        ):
            start_to_intermediate_rate = exchange_rates[(start_currency, intermediate_currency)]
            intermediate_to_target_rate = exchange_rates[(intermediate_currency, target_currency)]

            if start_to_intermediate_rate * intermediate_to_target_rate > 1:
                rate_of_return = start_to_intermediate_rate * intermediate_to_target_rate - 1

                if rate_of_return > max_rate_of_return:
                    max_rate_of_return = rate_of_return
                    optimal_path = (start_currency, intermediate_currency, target_currency, rate_of_return)

    if optimal_path:
        start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
        max_amount = start_amount
        # 최적 경로를 따라 환전
        max_amount *= exchange_rates[(start_currency, intermediate_currency)]
        max_amount *= exchange_rates[(intermediate_currency, end_currency)]

        # 직접 환전한 경우의 환율을 확인
        direct_conversion_rate = exchange_rates.get((start_currency, target_currency), 0)

        # 직접 환전한 경우의 화폐량 대비 수익률 계산
        direct_conversion_return = (max_amount / start_amount) - 1

        # 다른 가능한 경로와의 수익률을 계산
        other_paths = []
        for path in generate_all_exchange_paths(currencies, exchange_rates, start_currency, target_currency):
            if path != optimal_path:
                start_currency, intermediate_currency, end_currency, other_rate_of_return = path
                other_return = (max_amount / start_amount) - 1
                other_paths.append((intermediate_currency, end_currency, other_return))

        return max_amount, rate_of_return, direct_conversion_return, optimal_path, other_paths
    else:
        return start_amount, None, None, None, None


def generate_all_exchange_paths(currencies, exchange_rates, start_currency, target_currency):
    paths = []

    for intermediate_currency in currencies:
        if (
                start_currency != intermediate_currency
                and intermediate_currency != target_currency
                and (start_currency, intermediate_currency) in exchange_rates
                and (intermediate_currency, target_currency) in exchange_rates
        ):
            start_to_intermediate_rate = exchange_rates[(start_currency, intermediate_currency)]
            intermediate_to_target_rate = exchange_rates[(intermediate_currency, target_currency)]

            if start_to_intermediate_rate * intermediate_to_target_rate > 1:
                rate_of_return = start_to_intermediate_rate * intermediate_to_target_rate - 1
                paths.append((start_currency, intermediate_currency, target_currency, rate_of_return))

    return paths


start_currency = 'USD'
target_currency = 'JPY'

max_amount, rate_of_return, direct_conversion_return, optimal_path, other_paths = simulate_optimal_exchange_path(
    exchange_rates, start_currency, target_currency, start_amount=10000)

if optimal_path:
    start_currency, intermediate_currency, end_currency, rate_of_return = optimal_path
    print(f"Optimal Exchange Path: {start_currency} -> {intermediate_currency} -> {end_currency}")
    print(f"Maximum Exchange Amount: {max_amount:.4f} {end_currency}")
    print(f"Rate of Return (Compared to Direct Conversion): {rate_of_return:.4f}")
    print(f"Direct Conversion Rate of Return: {direct_conversion_return:.4f}")

    if other_paths:
        print("Other Possible Exchange Paths:")
        for path in other_paths:
            intermediate_currency, end_currency, other_rate_of_return = path
            print(
                f"{start_currency} -> {intermediate_currency} -> {end_currency} (Rate of Return: {other_rate_of_return:.4f})")
else:
    print(f"No optimal arbitrage opportunities found for {start_currency} -> {target_currency}.")
