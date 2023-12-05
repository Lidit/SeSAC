import os
import threading
import numpy as np

# from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.optim import SGD, Adam



# class Network(nn.Module):
#     lock = threading.Lock()
#
#     def __init__(self, input_dim=0, output_dim=0, lr=0.001,
#                 shared_network=None, activation='sigmoid', loss='mse'):
#
#         super(Network, self).__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.lr = lr
#         self.shared_network = shared_network
#         self.activation = activation
#         self.loss = loss
#         self.model = None
#
#     def predict(self, sample):
#         # with self.lock:
#         #     with graph.as_default():
#         #         if sess is not None:
#         #             set_session(sess)
#         #         return self.model.predict(sample).flatten()
#
#
#     def train_on_batch(self, x, y):
#         loss = 0.
#         # with self.lock:
#         #     with graph.as_default():
#         #         if sess is not None:
#         #             set_session(sess)
#         #         loss = self.model.train_on_batch(x, y)
#
#         return loss
#
#     def save_model(self, model_path):
#         if model_path is not None and self.model is not None:
#             self.model.save_weights(model_path, overwrite=True)
#
#     def load_model(self, model_path):
#         if model_path is not None:
#             self.model.load_weights(model_path)
#
#     @classmethod
#     def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
#         with graph.as_default():
#             if sess is not None:
#                 set_session(sess)
#
#             return LSTMNetwork.get_network_head(Input((num_steps, input_dim)))

# class LSTMNetwork(nn.Module):
# # class LSTMNetwork(Network):
#     def __init__(self, *args, num_steps=1, **kwargs):
#         super().__init__(*args, **kwargs)
#         with graph.as_default():
#             if sess is not None:
#                 set_session(sess)
#             self.num_steps = num_steps
#             inp = None
#             output = None
#             if self.shared_network is None:
#                 inp = Input((self.num_steps, self.input_dim))
#                 output = self.get_network_head(inp).output
#             else:
#                 inp = self.shared_network.input
#                 output = self.shared_network.output
#             output = Dense(
#                 self.output_dim, activation=self.activation,
#                 kernel_initializer='random_normal')(output)
#             self.model = Model(inp, output)
#             self.model.compile(
#                 optimizer=SGD(lr=self.lr), loss=self.loss)
#
#     @staticmethod
#     def get_network_head(inp):
#         output = LSTM(256, dropout=0.3,
#             return_sequences=True, stateful=False,
#             kernel_initializer='random_normal')(inp)
#         output = BatchNormalization()(output)
#         output = LSTM(128, dropout=0.3,
#             return_sequences=True, stateful=False,
#             kernel_initializer='random_normal')(output)
#         output = BatchNormalization()(output)
#         output = LSTM(64, dropout=0.3,
#             return_sequences=True, stateful=False,
#             kernel_initializer='random_normal')(output)
#         output = BatchNormalization()(output)
#         output = LSTM(32, dropout=0.3,
#             stateful=False,
#             kernel_initializer='random_normal')(output)
#         output = BatchNormalization()(output)
#         return Model(inp, output)
#
#     def train_on_batch(self, x, y):
#         x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
#         return super().train_on_batch(x, y)
#
#     def predict(self, sample):
#         sample = np.array(sample).reshape(
#             (1, self.num_steps, self.input_dim))
#         return super().predict(sample)



class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim, hidden_dim, n_layers):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        # self.hidden_size = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        # h0 = torch.zeros(self.n_layers, self.seq_len, self.hidden_dim)
        # c0 = torch.zeros(self.n_layers, self.seq_len, self.hidden_dim)
        out, _status = self.lstm(x)
        # out, _status = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.batch_norm(out[:, -1])
        out = self.fc(out)

        # out = out.view(-1, out.size(2))
        # out: (batch_size, output_size)
        return out

    @staticmethod
    def train_on_batch(model, x, y, optimizer, loss):
        outputs = model.forward(x)
        optimizer.zero_grad()
        loss = loss(outputs, y)
        loss.backward()
        optimizer.step()


class PolicyNetwork:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.model = LSTMNetwork(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, seq_len=5, n_layers=3)
        self.optimizer= SGD(model.parameters(), lr=lr)
        self.loss_function = nn.CrossEntropyLoss()
        self.prob = None

    def reset(self):
        self.prob = None

    def predict(self, sample):
        self.prob = self.model.forward(sample)
        return self.prob

    def train_on_batch(self, x, y):
        return model.train_on_batch(x=x, y=y, model=self.model, optimizer=self.optimizer, loss=self.loss_function)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            model_scripted = torch.jit.script(self.model)  # TorchScript 형식으로 내보내기
            model_scripted.save(model_path+'.pt')  # 저장하기

    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.jit.load(model_path+'.pt')
            self.model.eval()


if __name__ == '__main__':
    # from torchsummary import summary
    from torchinfo import summary

    model = LSTMNetwork(input_dim=5, output_dim=1, hidden_dim=256, n_layers=5, seq_len=5)
    # model = LSTMModel(7, 256, 5, 1)
    # x = torch.randn(64, 8)
    # out = model(x)
    # RuntimeError: For unbatched 2-D input, hx and cx should also be 2-D but got (3-D, 3-D) tensors

    x = torch.randn(5, 5, 5)
    out = model(x)  # works
    model.eval()
    # summary(model, input_data=x, mode='train', depth=3,
    #         col_names=["input_size", "output_size", "num_params", "mult_adds"],
    #         row_settings=["var_names"])
    policy = PolicyNetwork(input_dim=5, output_dim=1)
    summary(policy.model, input_data=x, mode='train', depth=3,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"])
