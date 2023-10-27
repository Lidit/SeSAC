# from forex_python.converter import CurrencyRates
#
# c = CurrencyRates()
# exchange_rate = c.get_rate('USD', 'KRW')  # USD to KRW
# print(exchange_rate)

import pandas as pd
import yfinance as yf

start = '2010-01-01'
end = '2022-12-31'
country = ['USD', 'KRW', "JPY", 'EUR', 'AUD', 'CNY','RUB', 'CHF', 'MXN']
ticker_list= []
for i in country:
    for j in country:
        if i ==j:
            continue
        ticker_list.append(i+j+"=X")

data = yf.download(ticker_list, start=start, end=end, interval='1mo')

print(data['Close'])

data['Close'].to_csv('exchange_rate.csv')