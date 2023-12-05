import logging
import os
# import settings
import data_manager

from learners import PolicyLearner

if __name__ == '__main__':
    stock_code = '005930'

    # log_dir = os.path.join('./project2', 'logs%s' % stock_code)

    chart_data = data_manager.load_chart_data('{}.csv'.format(stock_code))

    prep_data = data_manager.pre_process(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    # 미구현
    training_data.dropna()

    features_chart_data = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    chart_data = training_data[features_chart_data]

    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]

    training_data = training_data[features_training_data]


    policy_learner = PolicyLearner(stock_code=stock_code, chart_data=chart_data, training_data=training_data, min_trading_unit=1, max_traiding_unit=2, delayed_reward_threshold=.2, lr=.001)
    policy_learner.fit(balance=10000000, num_epoch=1000, discount_factor=0, start_epsilon=.5)

    model_dir = os.path.join('models/s' % stock_code)
    model_path = os.path.join(model_dir, 'models/s' % stock_code)
    policy_learner.policy_network.save_model(model_path)


