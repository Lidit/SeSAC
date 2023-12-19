# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from torch.utils.data import DataLoader, TensorDataset
# # # # # from sklearn.preprocessing import MinMaxScaler
# # # # # from sklearn.model_selection import train_test_split
# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # import matplotlib.pyplot as plt
# # # # #
# # # # # # CSV 파일 경로 설정 (실제 파일 경로에 맞게 수정)
# # # # # csv_file_path = "0.csv"
# # # # #
# # # # # # CSV 파일에서 데이터 로드
# # # # # df = pd.read_csv(csv_file_path, parse_dates=True, index_col="Date")
# # # # #
# # # # # # 다음 날 종가를 예측하기 위한 데이터 전처리 함수
# # # # # def create_multivariate_sequences(data, target_col, seq_length):
# # # # #     sequences, labels = [], []
# # # # #     for i in range(len(data) - seq_length):
# # # # #         seq = data.iloc[i:i+seq_length].values
# # # # #         label = data[target_col].iloc[i+seq_length]
# # # # #         sequences.append(seq)
# # # # #         labels.append(label)
# # # # #     return np.array(sequences), np.array(labels)
# # # # #
# # # # # # 데이터 정규화
# # # # # scaler = MinMaxScaler()
# # # # # scaled_data = scaler.fit_transform(df)
# # # # #
# # # # # # 시퀀스 길이 및 타겟 변수 선택
# # # # # sequence_length = 30  # 예: 30일간의 데이터를 기반으로 다음 날 종가를 예측
# # # # # target_variable = "Close"  # 종가를 예측
# # # # #
# # # # # # 시퀀스 및 레이블 생성
# # # # # X, y = create_multivariate_sequences(pd.DataFrame(scaled_data, columns=df.columns), target_variable, sequence_length)
# # # # #
# # # # # # Train 및 Test 세트로 분할
# # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # #
# # # # # # PyTorch로 데이터 로딩
# # # # # train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
# # # # # test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
# # # # # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# # # # # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# # # # #
# # # # # # 다변량 시계열 LSTM 모델 정의
# # # # # class MultivariateLSTM(nn.Module):
# # # # #     def __init__(self, input_size, hidden_size, output_size):
# # # # #         super(MultivariateLSTM, self).__init__()
# # # # #         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
# # # # #         self.fc = nn.Linear(hidden_size, output_size)
# # # # #
# # # # #     def forward(self, x):
# # # # #         _, (h_n, _) = self.lstm(x)
# # # # #         out = self.fc(h_n[-1, :, :])
# # # # #         return out
# # # # #
# # # # # # 모델 초기화
# # # # # input_size = len(df.columns)
# # # # # hidden_size = 50
# # # # # output_size = 1
# # # # # model = MultivariateLSTM(input_size, hidden_size, output_size)
# # # # #
# # # # # # 손실 함수 및 최적화 알고리즘 정의
# # # # # criterion = nn.MSELoss()
# # # # # optimizer = optim.Adam(model.parameters(), lr=0.001)
# # # # #
# # # # # # 학습
# # # # # num_epochs = 20
# # # # # for epoch in range(num_epochs):
# # # # #     for inputs, labels in train_loader:
# # # # #         optimizer.zero_grad()
# # # # #         outputs = model(inputs)
# # # # #         loss = criterion(outputs, labels.view(-1, 1))
# # # # #         loss.backward()
# # # # #         optimizer.step()
# # # # #
# # # # # # 평가
# # # # # model.eval()
# # # # # test_loss = 0.0
# # # # # with torch.no_grad():
# # # # #     for inputs, labels in test_loader:
# # # # #         outputs = model(inputs)
# # # # #         test_loss += criterion(outputs, labels.view(-1, 1)).item()
# # # # #
# # # # # print(f"Test Loss: {test_loss}")
# # # # #
# # # # # # 예측
# # # # # model.eval()
# # # # # predictions = []
# # # # # with torch.no_grad():
# # # # #     for inputs, _ in test_loader:
# # # # #         outputs = model(inputs)
# # # # #         predictions.extend(outputs.numpy())
# # # # #
# # # # # # 정규화 해제
# # # # # predictions = scaler.inverse_transform(predictions)
# # # # # y_test = scaler.inverse_transform(y_test)
# # # # #
# # # # # # 예측 결과 시각화
# # # # # plt.plot(y_test, label='True Prices')
# # # # # plt.plot(predictions, label='Predicted Prices')
# # # # # plt.legend()
# # # # # plt.show()
# # # # import torch
# # # # from sklearn.preprocessing import MinMaxScaler
# # # #
# # # # # 3차원 텐서 생성 (예시)
# # # # data = torch.randn((5, 5, 3))  # 5x5x3의 랜덤한 3차원 텐서 생성
# # # #
# # # # # MinMaxScaler를 사용하여 스케일링 (마지막 축을 기준으로)
# # # # scaler = MinMaxScaler()
# # # # scaled_data = torch.tensor(scaler.fit_transform(data.numpy().reshape(-1, data.shape[-1])))
# # # # scaled_data = scaled_data.view(data.shape)  # 다시 원래 모양으로 변환
# # # #
# # # # print("Original Data:")
# # # # print(data)
# # # # print("\nScaled Data:")
# # # # print(scaled_data)
# # # #
# # # # # 다시 역 스케일링 (마지막 축을 기준으로)
# # # # original_data = torch.tensor(scaler.inverse_transform(scaled_data.numpy().reshape(-1, data.shape[-1])))
# # # # original_data = original_data.view(data.shape)
# # # #
# # # # print("\nOriginal Data (After Descale):")
# # # # print(original_data)
# # # import torch
# # # import torch.nn as nn
# # # import yfinance as yf
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from torch.autograd import Variable
# # # import matplotlib.pyplot as plt
# # #
# # # # 데이터 불러오기
# # # data = yf.download('005930.KS', start='2022-01-01', end='2023-01-01')
# # # data = data[['Open', 'High', 'Low', 'Close']]
# # # data = data.astype('float32')
# # #
# # # # 데이터 스케일링
# # # scaler_x = StandardScaler()
# # # scaler_y = StandardScaler()
# # #
# # # X_ss = torch.tensor(scaler_x.fit_transform(data[['Open', 'High', 'Low', 'Close']].values))
# # # X_ss = X_ss.view(-1, 1, 4)  # 시퀀스 길이가 1인 3D 텐서로 변환
# # # y_ss = torch.tensor(scaler_y.fit_transform(data[['Open', 'High', 'Low', 'Close']].values))
# # # y_ss = y_ss.view(-1, 1, 4)
# # #
# # # # 학습 데이터와 테스트 데이터로 나누기
# # # X_train, X_test, y_train, y_test = train_test_split(X_ss, y_ss, test_size=0.2, shuffle=False)
# # #
# # # # 모델 정의
# # # class LSTMForecast(nn.Module):
# # #     def __init__(self, input_size, hidden_size, num_layers, output_size):
# # #         super(LSTMForecast, self).__init__()
# # #         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
# # #         self.linear = nn.Linear(in_features=hidden_size, out_features=128)
# # #         self.fc = nn.Linear(in_features=128, out_features=output_size)
# # #         self.relu = nn.ReLU()
# # #
# # #     def forward(self, x):
# # #         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()
# # #         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()
# # #         lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
# # #         hn = hn.view(-1, self.lstm.hidden_size)
# # #         out = self.relu(hn)
# # #         out = self.linear(out)
# # #         out = self.relu(out)
# # #         out = self.fc(out)
# # #         return out
# # #
# # # # 모델, 손실 함수, 최적화 함수 초기화
# # # input_size = 4
# # # hidden_size = 256
# # # num_layers = 1
# # # output_size = 4
# # # model = LSTMForecast(input_size, hidden_size, num_layers, output_size)
# # # loss_function = nn.MSELoss()
# # # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# # #
# # # # 시퀀스 길이를 5로 변경
# # # seq_length = 5
# # # X_train_seq = torch.cat([X_train[i:i+seq_length, :, :] for i in range(len(X_train)-seq_length+1)], dim=0)
# # # y_train_seq = torch.cat([y_train[i+seq_length:i+seq_length+1, :, :] for i in range(len(y_train)-seq_length+1)], dim=0)
# # #
# # # # 모델 학습
# # # num_epochs = 100
# # # for epoch in range(num_epochs):
# # #     output = model(X_train_seq)
# # #     optimizer.zero_grad()
# # #     loss = loss_function(output, y_train_seq)
# # #     loss.backward()
# # #     optimizer.step()
# # #
# # #     if epoch % 10 == 0:
# # #         print(f'Epoch:{epoch}, loss:{loss.item():1.5f}')
# # #
# # # # 예측
# # # df_x_ss = torch.tensor(scaler_x.transform(data[['Open', 'High', 'Low', 'Close']].values))
# # # df_x_ss = df_x_ss.view(-1, 1, 4)
# # #
# # # train_predict = model(df_x_ss)
# # # predicted = train_predict.data.numpy()
# # # predicted = scaler_y.inverse_transform(predicted.reshape(-1, output_size))
# # # label_y = scaler_y.inverse_transform(y_ss.numpy().reshape(-1, output_size))
# # #
# # # # 시각화
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(label_y[:, 0], label='Actual Open')
# # # plt.plot(label_y[:, 1], label='Actual High')
# # # plt.plot(label_y[:, 2], label='Actual Low')
# # # plt.plot(label_y[:, 3], label='Actual Close')
# # # plt.plot(predicted[:, 0], label='Predicted Open', linestyle='--')
# # # plt.plot(predicted[:, 1], label='Predicted High', linestyle='--')
# # # plt.plot(predicted[:, 2], label='Predicted Low', linestyle='--')
# # # plt.plot(predicted[:, 3], label='Predicted Close', linestyle='--')
# # # plt.title('Forecasting')
# # # plt.legend()
# # # plt.show()
# # #
# # # import torch
# # # import torch.nn as nn
# # # import numpy as np
# # # import pandas as pd
# # # from sklearn.preprocessing import MinMaxScaler
# # # from torch.autograd import Variable
# # # import matplotlib.pyplot as plt
# # #
# # # # Many-to-Many LSTM 모델 정의
# # # class StockLSTM(nn.Module):
# # #     def __init__(self, input_size, hidden_size, output_size, num_layers):
# # #         super(StockLSTM, self).__init__()
# # #         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
# # #         self.linear = nn.Linear(hidden_size, output_size)
# # #
# # #     def forward(self, x):
# # #         lstm_out, _ = self.lstm(x)
# # #         output = self.linear(lstm_out)
# # #         return output
# # #
# # # # 주식 데이터 로딩 및 전처리
# # # # 이 예시에서는 yfinance 라이브러리를 사용하여 주식 데이터를 가져옵니다.
# # # import yfinance as yf
# # #
# # # # 주식 데이터 다운로드 (예시로 삼성전자 주식 데이터 사용)
# # # stock_data = yf.download('005930.KS', start='2022-01-01', end='2023-01-01')
# # #
# # # # 필요한 피처 선택 (OHLCV)
# # # stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
# # #
# # # # 데이터 정규화
# # # scaler = MinMaxScaler()
# # # scaled_data = scaler.fit_transform(stock_data)
# # #
# # # # 학습 데이터 생성 함수
# # # def create_sequences(data, seq_length):
# # #     sequences = []
# # #     for i in range(len(data) - seq_length):
# # #         seq = data[i:i + seq_length, :]
# # #         label = data[i + seq_length:i + seq_length + 1, :]
# # #         sequences.append((seq, label))
# # #     return sequences
# # #
# # # # 학습 데이터 생성
# # # seq_length = 10  # 시퀀스 길이 설정
# # # data_sequences = create_sequences(scaled_data, seq_length)
# # #
# # # # 데이터를 PyTorch 텐서로 변환
# # # X = torch.tensor([seq for seq, _ in data_sequences], dtype=torch.float32)
# # # y = torch.tensor([label for _, label in data_sequences], dtype=torch.float32)
# # #
# # # # 모델 및 학습 설정
# # # input_size = 5  # OHLCV의 피처 수
# # # hidden_size = 64
# # # output_size = 5  # 출력은 다시 OHLCV
# # # num_layers = 2
# # # learning_rate = 0.001
# # #
# # # model = StockLSTM(input_size, hidden_size, output_size, num_layers)
# # # criterion = nn.MSELoss()
# # # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # #
# # # # 학습 루프
# # # num_epochs = 50
# # # losses = []  # 손실 기록을 위한 리스트 추가
# # # for epoch in range(num_epochs):
# # #     # Forward pass
# # #     outputs = model(X)
# # #
# # #     # 손실 계산
# # #     loss = criterion(outputs, y)
# # #     losses.append(loss.item())  # 손실 기록 추가
# # #
# # #     # Backward pass 및 옵티마이저 업데이트
# # #     optimizer.zero_grad()
# # #     loss.backward()
# # #     optimizer.step()
# # #
# # #     if epoch % 10 == 0:
# # #         print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
# # #
# # # # 손실 그래프 시각화
# # # plt.plot(losses, label='Training Loss')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Loss')
# # # plt.title('Training Loss Over Time')
# # # plt.legend()
# # # plt.show()
# # #
# # # # 향후 1년치 데이터 예측
# # # future_input_sequence = torch.randn(1, seq_length, input_size)
# # # predicted_sequence = model(future_input_sequence)
# # #
# # # # 정규화 해제
# # # predicted_sequence = scaler.inverse_transform(predicted_sequence.squeeze().detach().numpy())
# # #
# # # # 실제 데이터와 예측 데이터 시각화
# # # plt.figure(figsize=(10, 6))
# # # for i in range(output_size):
# # #     plt.plot(scaled_data[:, i], label=f'Actual {stock_data.columns[i]}', alpha=0.7)
# # #     plt.plot(np.arange(len(scaled_data), len(scaled_data) + seq_length),
# # #              predicted_sequence[:, i], label=f'Predicted {stock_data.columns[i]}', linestyle='--', alpha=0.7)
# # #
# # # plt.xlabel('Time')
# # # plt.ylabel('Normalized Value')
# # # plt.title('Actual vs Predicted Stock Prices')
# # # plt.legend()
# # # plt.show()
# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import MinMaxScaler
# # from torch.autograd import Variable
# # import matplotlib.pyplot as plt
# #
# # # Many-to-Many LSTM 모델 정의
# # class StockLSTM(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size, num_layers):
# #         super(StockLSTM, self).__init__()
# #         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
# #         self.linear = nn.Linear(hidden_size, output_size)
# #
# #     def forward(self, x):
# #         lstm_out, _ = self.lstm(x)
# #         output = self.linear(lstm_out)
# #         return output
# #
# # # 주식 데이터 로딩 및 전처리
# # # 이 예시에서는 yfinance 라이브러리를 사용하여 주식 데이터를 가져옵니다.
# # import yfinance as yf
# #
# # # 주식 데이터 다운로드 (예시로 삼성전자 주식 데이터 사용)
# # stock_data = yf.download('005930.KS', start='2022-01-01', end='2023-01-01')
# #
# # # 필요한 피처 선택 (OHLCV)
# # stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
# #
# # # 데이터 정규화
# # scaler = MinMaxScaler()
# # scaled_data = scaler.fit_transform(stock_data)
# #
# # # 학습 데이터 생성 함수
# # def create_sequences(data, seq_length):
# #     sequences = []
# #     for i in range(len(data) - seq_length):
# #         seq = data[i:i + seq_length, :]
# #         label = data[i + seq_length:i + seq_length + 1, :]
# #         sequences.append((seq, label))
# #     return sequences
# #
# # # 학습 데이터 생성
# # seq_length = 10  # 시퀀스 길이 설정
# # data_sequences = create_sequences(scaled_data, seq_length)
# #
# # # 데이터를 PyTorch 텐서로 변환
# # X = torch.tensor([seq for seq, _ in data_sequences], dtype=torch.float32)
# # y = torch.tensor([label for _, label in data_sequences], dtype=torch.float32)
# #
# # # 모델 및 학습 설정
# # input_size = 5  # OHLCV의 피처 수
# # hidden_size = 64
# # output_size = 5  # 출력은 다시 OHLCV
# # num_layers = 2
# # learning_rate = 0.001
# #
# # model = StockLSTM(input_size, hidden_size, output_size, num_layers)
# # criterion = nn.MSELoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# #
# # # 학습 루프
# # num_epochs = 50
# # losses = []  # 손실 기록을 위한 리스트 추가
# # for epoch in range(num_epochs):
# #     # Forward pass
# #     outputs = model(X)
# #
# #     # 손실 계산
# #     loss = criterion(outputs, y)
# #     losses.append(loss.item())  # 손실 기록 추가
# #
# #     # Backward pass 및 옵티마이저 업데이트
# #     optimizer.zero_grad()
# #     loss.backward()
# #     optimizer.step()
# #
# #     if epoch % 10 == 0:
# #         print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
# #
# # # 손실 그래프 시각화
# # plt.plot(losses, label='Training Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.title('Training Loss Over Time')
# # plt.legend()
# # plt.show()
# #
# # # 향후 1년치 데이터 예측
# # future_input_sequence = torch.randn(1, seq_length, input_size)
# # predicted_sequence = model(future_input_sequence)
# #
# # # 정규화 해제
# # predicted_sequence = scaler.inverse_transform(predicted_sequence.squeeze().detach().numpy())
# # stock_data = stock_data.reset_index(drop=True)
# # # predicted_sequence= predicted_sequence.reset_index()
# # # 실제 데이터와 예측 데이터 시각화
# # plt.figure(figsize=(10, 6))
# # for i in range(output_size):
# #     plt.plot(stock_data.iloc[-len(predicted_sequence):, i], label=f'Actual {stock_data.columns[i]}', alpha=0.7)
# #     plt.plot(predicted_sequence[:, i], label=f'Predicted {stock_data.columns[i]}', linestyle='--', alpha=0.7)
# #
# # plt.xlabel('Time')
# # plt.ylabel('Stock Price')
# # plt.title('Actual vs Predicted Stock Prices')
# # plt.legend()
# # plt.show()
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from torch.autograd import Variable
#
# # 데이터 로드 및 전처리
# data = pd.read_csv('0.csv')
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)
# data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
#
# # 데이터 정규화
# scaler = MinMaxScaler(feature_range=(-1, 1))
# data_normalized = scaler.fit_transform(data)
#
# # 시퀀스 생성 함수
# def create_sequences(data, seq_length, target_steps):
#     sequences = []
#     for i in range(len(data) - seq_length - target_steps + 1):
#         seq = data[i:i + seq_length]
#         label = data[i + seq_length:i + seq_length + target_steps]
#         sequences.append((seq, label))
#     return sequences
#
# # 하이퍼파라미터
# seq_length = 10
# target_steps = 5
# input_size = 5  # 특성의 수
# hidden_size = 64
# output_size = 5
# learning_rate = 0.001
# num_epochs = 100
#
# # 데이터를 시퀀스로 변환
# sequences = create_sequences(data_normalized, seq_length, target_steps)
# X = torch.tensor([seq for seq, _ in sequences], dtype=torch.float32)
# y = torch.tensor([label for _, label in sequences], dtype=torch.float32)
#
# # 모델 정의
# class MultiStepLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, target_steps):
#         super(MultiStepLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size * target_steps)
#
#     def forward(self, x):
#         _, (hidden, _) = self.lstm(x)
#         out = self.fc(hidden.view(-1, hidden_size))
#         out = out.view(-1, target_steps, output_size)
#         return out
#
# model = MultiStepLSTM(input_size, hidden_size, output_size, target_steps)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 학습
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = model(X)
#     loss = criterion(output, y)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # 예측
# model.eval()
# with torch.no_grad():
#     future = 30  # 향후 30일 예측
#     current_sequence = data_normalized[-seq_length:].tolist()
#     predicted = []
#
#     for i in range(future):
#         input_sequence = torch.tensor(current_sequence[-seq_length:], dtype=torch.float32).view(1, seq_length, input_size)
#         prediction = model(input_sequence)
#         current_sequence.append(prediction[-1].tolist())
#         predicted.append(scaler.inverse_transform(np.array(prediction[-1]).reshape(-1, 5)).tolist())
#
# # 결과 시각화
# import matplotlib.pyplot as plt
#
# actual = scaler.inverse_transform(y[-future:].numpy().reshape(-1, 5))
# predicted = np.array(predicted).reshape(-1, 5)
#
# plt.figure(figsize=(12, 6))
# for i in range(5):  # 특성의 수
#     plt.subplot(2, 3, i+1)
#     plt.plot(actual[:, i], label=f'Actual {data.columns[i]}')
#     plt.plot(predicted[:, i], label=f'Predicted {data.columns[i]}', linestyle='--')
#     plt.title(f'{data.columns[i]} Prediction')
#     plt.xlabel('Days')
#     plt.ylabel(f'{data.columns[i]} Value')
#     plt.legend()
#
# plt.tight_layout()
# plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import pandas_datareader as pdr
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from torch.autograd import Variable

# 데이터 로드 및 전처리
# data = pd.read_csv('0.csv')
data = pdr.get_data_yahoo('005380.KS', '2010-01-01', '2022-12-31')
# print(data)
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(data)

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data)

# 시퀀스 생성 함수
def create_sequences(data, seq_length, target_steps):
    sequences = []
    for i in range(len(data) - seq_length - target_steps + 1):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length + target_steps]
        sequences.append((seq, label))
    return sequences

# 하이퍼파라미터
seq_length = 30
target_steps = 10
input_size = 5  # 특성의 수
hidden_size = 64
output_size = 5
learning_rate = 0.001
num_epochs = 100

# 데이터를 시퀀스로 변환
sequences = create_sequences(data_normalized, seq_length, target_steps)
X = torch.tensor([seq for seq, _ in sequences], dtype=torch.float32)
y = torch.tensor([label for _, label in sequences], dtype=torch.float32)

# 모델 정의
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, target_steps):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * target_steps)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.view(-1, hidden_size))
        out = out.view(-1, target_steps, output_size)
        return out

model = MultiStepLSTM(input_size, hidden_size, output_size, target_steps)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
model.eval()
with torch.no_grad():
    future = 1
    # 향후 30일 예측
    current_sequence = data_normalized[-seq_length:]
    predicted = []

    for i in range(future):
        input_sequence = torch.tensor(current_sequence[-seq_length:], dtype=torch.float32).view(1, seq_length, input_size)
        prediction = model(input_sequence)
        current_sequence = np.concatenate([current_sequence, prediction[-1].numpy()])
        predicted.append(scaler.inverse_transform(np.array(prediction[-1]).reshape(-1, 5)).tolist())

# 결과 시각화
import matplotlib.pyplot as plt

actual = scaler.inverse_transform(y[-future:].numpy().reshape(-1, 5))
predicted = np.array(predicted).reshape(-1, 5)

plt.figure(figsize=(12, 6))
# for i in range(1):  # 특성의 수
#     # plt.subplot(2, 3, i+1)
#     plt.plot(actual[:, i], label=f'Actual {data.columns[i]}')
#     plt.plot(predicted[:, i], label=f'Predicted {data.columns[i]}', linestyle='--')
#     plt.title(f'{data.columns[i]} Prediction')
#     plt.xlabel('Days')
#     plt.ylabel(f'{data.columns[i]} Value')
#     plt.legend()
    # plt.subplot(2, 3, i+1)
plt.plot(actual[:, 3], label=f'Actual {data.columns[2]}')
plt.plot(predicted[:, 3], label=f'Predicted {data.columns[2]}', linestyle='--')
plt.title(f'{data.columns[3]} Prediction')
plt.xlabel('Days')
plt.ylabel(f'{data.columns[3]} Value')
plt.legend()

plt.tight_layout()
plt.show()

def calculate_mape(predictions, targets):
    return np.mean(np.abs((targets - predictions) / targets)) * 100


mae = calculate_mape(predicted[:,3], actual[:,3])
print(f'Mean Absolute Error (MAPE) on Test Data: {mae:.4f}')
print(predicted[:,3])
print(actual[:,3])