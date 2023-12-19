import numpy as np
import pandas as pd
import os

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']
COLUMNS_TRAINING_DATA_V1 = [
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
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
    return chart_data


def pre_process(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]

    for window in windows:
        prep_data[f'close_ma{window}'] = prep_data['close'].rolling(window).mean()
        prep_data[f'volume_ma{window}'] = prep_data['volume'].rolling(window).mean()

    return prep_data


def build_training_data(prep_data):
    training_data = prep_data

    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['open_lastclose_ratio'].iloc[1:] = (
            (training_data['open'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
    )

    training_data['high_close_ratio'] = (
            (training_data['high'].values - training_data['close'].values) / training_data['close'].values)

    training_data['low_close_ratio'] = (
            (training_data['low'].values - training_data['close'].values) / training_data['close'].values
    )

    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['close_lastclose_ratio'].iloc[1:] = (
            (training_data['close'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
    )
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data['volume_lastvolume_ratio'].iloc[1:] = (
            (training_data['volume'][1:].values - training_data['volume'][:-1].values) /
            training_data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0,method='bfill').values
    )

    windows = [5, 10, 20, 60, 120]

    for window in windows:
        training_data['close_ma%d_ratio' % window] = (
                (training_data['close'] - training_data['close_ma%d' % window]) / training_data['close_ma%d' % window]
        )
        training_data['volume_ma%d_ratio' % window] = (
                (training_data['volume'] - training_data['volume_ma%d' % window]) / training_data['volume_ma%d' % window]
        )

    return training_data


def load_data(code, date_from, date_to):
    # if ver in ['v3', 'v4']:
    #     return load_data_v3_v4(code, date_from, date_to, ver)

    # header = None if ver == 'v1' else 0
    header = None
    df = pd.read_csv(
        os.path.join(f'{code}.csv'),
        thousands=',', header=0, converters={'date': lambda x: str(x)})

    # if ver == 'v1':
    #     df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = pre_process(df)

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    # if ver == 'v1':
    #     training_data = df[COLUMNS_TRAINING_DATA_V1]
    # elif ver == 'v1.1':
    #     training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    # elif ver == 'v2':
    #     df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
    #     training_data = df[COLUMNS_TRAINING_DATA_V2]
    #     training_data = training_data.apply(np.tanh)
    # else:
    #     raise Exception('Invalid version.')
    # training_data = df[COLUMNS_TRAINING_DATA_V1]
    training_data = build_training_data(df)

    return chart_data, training_data
