import torch
import torch.nn as nn
from torchinfo import summary


# 바닐?라 lstm 모델을 설계 해 보았어요
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        """
        input_size(input_dim) = input x에 대한 features의 수 (데이터 칼럼 수)
        hidden_size(hidden_dim) = hidden state의 features의 수
        n_layers = lstm의 레이어 수
        batch_first: 입출력 텐서의 형태가 다음과 같음.기본값은False
                        True로 설정시(batch, seq, feature)
                        False로설정시(seq, batch, feature)
        """
        super(LSTMForecast, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers,
                            dropout=0.3,
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        param
            x(torch.Tensor): input 의 shape 는 (batch size, sequence length, input_dim)
        return:
            torch.Tensor: output은 (batch size, output_dim)의 shape를 가진 값 이에요
        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn.view(-1, self.lstm.hidden_size)
        out = self.relu(hn)
        out = self.linear(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)

        # lstm_out, _ = self.lstm(x)
        # out = self.linear(lstm_out[:, -1, :])
        # out = self.relu(out)
        return out


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    from torch.autograd import Variable
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('005930.KS.csv')

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Volume'] = data['Volume'].astype('float32')
    print(data.shape)
    print(data.dtypes)
    data = data.astype('float32')
    print(data.dtypes)

    # print(len(data))
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length*2):
            seq = data.iloc[i:i + seq_length].values
            label = data.iloc[i + seq_length:i + seq_length*2].values
            sequences.append((seq, label))
        return sequences


    seq_length = 30
    x = data.loc[:, ['Open', 'High', 'Low', 'Close']]
    sequences = create_sequences(x, seq_length)
    print(len(sequences))

    x = torch.tensor([seq for seq, _ in sequences], dtype=torch.float32).view(-1, seq_length, 4)
    y = torch.tensor([label for _, label in sequences], dtype=torch.float32).view(-1, seq_length, 4)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_ss = torch.tensor(x_scaler.fit_transform(x.numpy().reshape(-1, x.shape[-1])))
    X_ss = X_ss.view(x.shape)
    y_ss = torch.tensor(y_scaler.fit_transform(y.numpy().reshape(-1, y.shape[-1])))
    y_ss = y_ss.view(y.shape)
    print(X_ss.shape, y_ss.shape)
    X_train = X_ss[:110, :]
    X_test = X_ss[110:, :]

    y_train = y_ss[:110, :]
    y_test = y_ss[110:, :]

    print(X_train)
    print(y_train)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # print(X_test)
    # print(type(y_test))

    # y_test_descale = torch.tensor(y_scaler.inverse_transform(y_test.numpy().reshape(-1, y_test.shape[-1])))
    # y_test_descale = y_test_descale.view(y_test.shape)
    # y_test_descale = y_test_descale.numpy()
    #
    # X_train_tensors = Variable(torch.tensor(X_train))
    # X_test_tensors = Variable(torch.tensor(X_test))
    #
    # y_train_tensors = Variable(torch.tensor(y_train))
    # y_test_tensors = Variable(torch.tensor(y_test))
    #
    # # X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]), )
    # # X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
    #
    # print(X_train_tensors.shape, y_train_tensors.shape)
    # print(X_test_tensors.shape, y_test_tensors.shape)
    #
    # num_epochs = 300
    # lr = 0.002
    #
    # input_size = 4
    # hidden_size = 128
    # num_layers = 2
    #
    # output_size = 4
    #
    # model = LSTMForecast(input_size, hidden_size, num_layers, output_size)
    #
    # loss_function = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # for epoch in range(num_epochs):
    #     output = model.forward(X_train_tensors)
    #     optimizer.zero_grad()
    #     loss = loss_function(output, y_train_tensors)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if epoch % 10 == 0:
    #         print(f'Epoch:{epoch}, loss:{loss.item():1.5f}')
    #
    # # x = torch.tensor([seq for seq, _ in sequences], dtype=torch.float32).view(-1, seq_length, 4)
    # # y = torch.tensor([label for _, label in sequences], dtype=torch.float32).view(-1, 1, 4)
    #
    # # df_x_ss = torch.tensor([seq for seq, _ in sequences], dtype=torch.float32).view(-1, seq_length, 4)
    # # df_y_ss = torch.tensor([label for _, label in sequences], dtype=torch.float32).view(-1, 1, 4)
    # # print(df_x_ss.shape)
    # # print(df_y_ss.shape)
    # #
    # # df_x_ms = torch.tensor(x_scaler.fit_transform(df_x_ss.numpy().reshape(-1, df_x_ss.shape[-1])))
    # # df_x_ms = df_x_ms.view(df_x_ss.shape)
    # # print(df_x_ms.shape)
    # #
    # # df_x_ss = Variable(torch.Tensor(df_x_ss))
    # # df_y_ss = Variable(torch.Tensor(df_y_ss))
    #
    # # df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))
    # model.eval()
    # with torch.no_grad():
    #     train_predict = model.forward(X_test_tensors)
    # print(train_predict.shape)
    # train_predict = train_predict.detach()
    # predicted = torch.tensor(y_scaler.inverse_transform(train_predict.reshape(-1, train_predict.shape[-1])))
    # predicted = predicted.view(train_predict.shape)
    # predicted = predicted.detach().numpy()
    # predicted = predicted.reshape(-1, 4)
    #
    # # predicted = y_scaler.inverse_transform(predicted)
    #
    #
    # # label_y = df_y_ss.numpy()
    # label_y = y_test_descale
    #
    # # print(predicted)
    #
    # # label_y = y_scaler.inverse_transform(label_y.reshape(-1, 4))
    # label_y = label_y.reshape(-1, 4)
    # # print(label_y)
    # # label_y = torch.tensor(ss.inverse_transform(label_y.reshape(-1, label_y.shape[-1])))
    # # label_y = label_y.view(df_y_ss.shape)
    # # label_y = label_y.numpy()
    # print(label_y[0:3])
    # print(predicted[0:3])
    #
    # plt.figure(figsize=(10, 6))
    # # plt.axvline(x=200, c='r', linestyle='--')
    #
    # plt.plot(label_y[:, 0], label='Actual Open')
    # plt.plot(label_y[:, 1], label='Actual High')
    # plt.plot(label_y[:, 2], label='Actual Low')
    # plt.plot(label_y[:, 3], label='Actual Close')
    # plt.plot(predicted[:, 0], label='Predicted Open', linestyle='--')
    # plt.plot(predicted[:, 1], label='Predicted High', linestyle='--')
    # plt.plot(predicted[:, 2], label='Predicted Low', linestyle='--')
    # plt.plot(predicted[:, 3], label='Predicted Close', linestyle='--')
    # plt.title('Forecasting')
    # plt.legend()
    # plt.show()
