import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터셋 가져오기
#origin_df = pdr.get_data_yahoo('005380.KS', '2010-01-01', '2021-12-31')
# origin_df = pdr.get_data_yahoo('AAPL', '2010-01-01', '2023-12-31')
# origin_df = pdr.get_data_yahoo('003550.KS', '2010-01-01', '2023-12-31')
origin_df = pdr.get_data_yahoo('038950.KQ', '2011-01-01', '2021-12-31')

# 지표 가져오기
origin_df['Date'] = origin_df.index
origin_df['YearQuarter'] = origin_df.index.to_period('Q')
origin_df['YearQuarter'] = origin_df['YearQuarter'].astype(str).apply(lambda x: x[:4] + '-' + x[4:])

# RSI 재무제표
index_data1 = pd.read_csv('RSI재무제표.csv')
index_data1 = index_data1.transpose()
index_data1.columns = index_data1.iloc[0]
index_data1['YearQuarter'] = index_data1.index

df = pd.merge(origin_df,
              index_data1[['YearQuarter', 'RSI ROA', 'RSI ROE']], on='YearQuarter')

# 지수단위통합
index_data2 = pd.read_csv('자료조사1차_지수단위통합_20231215_JHD.csv', encoding='euc-kr')
index_data2['날짜'] = pd.to_datetime(index_data2['날짜'], format='%m/%d/%Y')
index_data2.set_index('날짜', inplace=True)
index_data2['YearQuarter'] = index_data2.index.to_period('Q')
index_data2['YearQuarter'] = index_data2['YearQuarter'].astype(str).apply(lambda x: x[:4] + '-' + x[4:])

df = pd.merge(df, index_data2, on='YearQuarter')

# 인지도 지수
index_data3 = pd.read_csv('인지도.csv')
index_data3 = index_data3.transpose()
index_data3.columns = index_data3.iloc[0]
index_data3['YearQuarter'] = index_data3.index

df = pd.merge(df, index_data3[['YearQuarter', 'RSI 인지도']], on='YearQuarter')
df.fillna(0, inplace=True)
df.set_index('Date', inplace=True)

#df = origin_df

df = df.reset_index()
df['Date'] = df['Date'].astype('object')

# Date 를 index로 변경
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Volume'] = df['Volume'].astype(float)

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI ROA', 'RSI ROE',
        '경제 신뢰 지수', '소비자 심리 지수',
        'KOSPI return rate', 'GDP growth', 'RSI 인지도']]  # OHLCV 데이터 사용
y = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # OHLCV를 예측 대상으로 선택

ms = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_ms = ss.fit_transform(y)

# 2011 ~ 2021 데이터셋 7:3 기준 인덱스 계산
split_index = int(len(X_ss) * 0.7)

X_train = X_ss[:split_index, :]
X_test = X_ss[split_index:, :]

y_train = y_ms[:split_index, :]
y_test = y_ms[split_index:, :]

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train)).to(device)  # GPU로 이동
X_test_tensors = Variable(torch.Tensor(X_test)).to(device)    # GPU로 이동

y_train_tensors = Variable(torch.Tensor(y_train)).to(device)  # GPU로 이동
y_test_tensors = Variable(torch.Tensor(y_test)).to(device)    # GPU로 이동

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_f.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_f.shape, y_test_tensors.shape)

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, output_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # GPU로 이동
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # GPU로 이동

        output, _ = self.lstm(x, (h_0, c_0))
        out = self.relu(output)
        out = self.fc(out[:, -1, :])  # Using only the last output
        return out

num_epochs = 10000
learning_rate = 0.001

num_features = 12
hidden_size = 100
output_size = 5
num_layers = 1

model = LSTM(num_features, hidden_size, output_size, num_layers, X_train_tensors_f.shape[1]).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(X_train_tensors_f.to(device))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors.to(device))

    loss.backward()
    optimizer.step()

    if epoch % 999 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# 결과 시각화
with torch.no_grad():
    train_predict = model(X_train_tensors_f.to(device))
    test_predict = model(X_test_tensors_f.to(device))

train_predict = ss.inverse_transform(train_predict.cpu().numpy())
test_predict = ss.inverse_transform(test_predict.cpu().numpy())
y_train = ss.inverse_transform(y_train)
y_test = ss.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
# plt.axvline(x=2900, c='r', linestyle='--')

# plt.plot(y_test, label='Actual Data (Test)')
# plt.plot(test_predict, label='Predicted Data (Test)')

# plt.plot(train_predict[:,0:4], label='Actual Data (Test)')
# plt.plot(y_train[:,0:4], label='Predicted Data (Test)')

plt.plot(test_predict[-252:,0:4], label='Actual Data (Test)')
plt.plot(y_test[-252:,0:4], label='Predicted Data (Test)', linestyle='--')

plt.title('Time-Series Prediction (Test Data)')
plt.legend()

plt.savefig('LG 1-1.png')
# 모델 저장
torch.save(model.state_dict(), 'lstm_model-LG.pth')

plt.show()