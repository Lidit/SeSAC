import numpy as np
import pandas as pd
import os

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']
COLUMNS_TRAINING_DATA = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]


def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',')
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data


# def pre_process(chart_data):
#     prep_data = chart_data
#     windows = [5, 10, 20, 60, 120]
#
#     for window in windows:
#         prep_data[f'close_ma{window}'] = prep_data['close'].rolling(window).mean()
#         prep_data[f'volume_ma{window}'] = prep_data['volume'].rolling(window).mean()
#
#     return prep_data
#
#
# def build_training_data(prep_data):
#     training_data = prep_data
#
#     training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
#     training_data['open_lastclose_ratio'].iloc[1:] = (
#             (training_data['open'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
#     )
#
#     training_data['high_close_ratio'] = (
#             (training_data['high'].values - training_data['close'].values) / training_data['close'].values)
#
#     training_data['low_close_ratio'] = (
#             (training_data['low'].values - training_data['close'].values) / training_data['close'].values
#     )
#
#     training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
#     training_data['close_lastclose_ratio'].iloc[1:] = (
#             (training_data['close'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
#     )
#     training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
#     training_data['volume_lastvolume_ratio'].iloc[1:] = (
#             (training_data['volume'][1:].values - training_data['volume'][:-1].values) /
#             training_data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0,method='bfill').values
#     )
#
#     windows = [5, 10, 20, 60, 120]
#
#     for window in windows:
#         training_data['close_ma%d_ratio' % window] = (
#                 (training_data['close'] - training_data['close_ma%d' % window]) / training_data['close_ma%d' % window]
#         )
#         training_data['volume_ma%d_ratio' % window] = (
#                 (training_data['volume'] - training_data['volume_ma%d' % window]) / training_data['volume_ma%d' % window]
#         )
#
#     return training_data
def pre_process(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
            (data['volume'][1:].values - data['volume'][:-1].values)
            / data['volume'][:-1].replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values
    )
    print(data)
    return data


def load_data(code, date_from, date_to):

    header = None
    df = pd.read_csv(
        os.path.join(f'{code}.csv'),
        thousands=',', header=0, converters={'date': lambda x: str(x)})

    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = pre_process(df)
    # print(df.head())
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)
    print("분리전\n", df.head())
    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]
    print("분리후 차트데이터\n", chart_data.head())
    # 학습 데이터 분리
    training_data = df[COLUMNS_TRAINING_DATA]

    return chart_data, training_data
