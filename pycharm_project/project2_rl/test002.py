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


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length * 2):
        seq = data.iloc[i:i + seq_length].values
        label = data.iloc[i + seq_length:i + seq_length * 2].values
        sequences.append((seq, label))
    return sequences


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 데이터셋 가져오기
origin_df = pdr.get_data_yahoo('005380.KS', '2010-01-01', '2022-12-31')
df = origin_df

df = df.reset_index()
df['Date'] = df['Date'].astype('object')

# Date 를 index로 변경
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Volume'] = df['Volume'].astype(float)

X = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # OHLCV 데이터 사용
y = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # OHLCV를 예측 대상으로 선택
ms = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_ms = ss.fit_transform(y)

seq = create_sequences(X, 30)
x = torch.tensor([seq for seq, _ in seq], dtype=torch.float32).view(-1, 30, 5)
y = torch.tensor([label for _, label in seq], dtype=torch.float32).view(-1, 30, 5)

print(x.shape, y.shape)


# X_train = X_ss[:2900, :]
# X_test = X_ss[2900:, :]
#
# y_train = y_ms[:2900, :]
# y_test = y_ms[2900:, :]

X_train = x[:2900]
X_test = x[2900:]

y_train = y[:2900]
y_test = y[2900:]

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

X_train_tensors = Variable(torch.Tensor(X_train)).to(device)  # GPU로 이동
X_test_tensors = Variable(torch.Tensor(X_test)).to(device)    # GPU로 이동

y_train_tensors = Variable(torch.Tensor(y_train)).to(device)  # GPU로 이동
y_test_tensors = Variable(torch.Tensor(y_test)).to(device)    # GPU로 이동

# X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 30, X_train_tensors.shape[1]))
# X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 30, X_test_tensors.shape[1]))

# print("Training Shape", X_train_tensors_f.shape, y_train_tensors.shape)
# print("Testing Shape", X_test_tensors_f.shape, y_test_tensors.shape)

class LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # GPU로 이동
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # GPU로 이동

        output, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.relu(output)
        out = self.fc(out)  # Using only the last output
        return out

num_epochs = 1000
learning_rate = 0.001

num_features = 5
hidden_size = 100
num_layers = 1

model = LSTM(num_features, hidden_size, num_layers, X_train_tensors.shape[1]).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(X_train_tensors.to(device))
    # print(outputs.shape)
    # print(y_train_tensors.shape)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors.to(device))

    loss.backward()
    optimizer.step()

    if epoch % 999 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# 결과 시각화
with torch.no_grad():
    train_predict = model(X_train_tensors.to(device))
    test_predict = model(X_test_tensors.to(device))

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

plt.plot(test_predict[:,0:4], label='Actual Data (Test)')
plt.plot(y_test[:,0:4], label='Predicted Data (Test)', linestyle='--')

plt.title('Time-Series Prediction (Test Data)')
plt.legend()
plt.show()

# plt.savefig('DeepLearning_pytorch/my/chap07/LSTM_STOCK_DATA/1-1.png')
# 모델 저장
# torch.save(model.state_dict(), 'DeepLearning_pytorch/my/chap07/LSTM_STOCK_DATA/lstm_model.pth')