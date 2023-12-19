import torch
import torch.nn as nn


class ForecastingStockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForecastingStockRNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).requires_grad_()
        rnn_out, _ = self.rnn(x, h0.detach())
        out = self.linear(rnn_out[:, -1])
        return out


if __name__ == '__main__':
    from torchinfo import summary

    model = ForecastingStockRNN(6,64, 1)

    x = torch.randn(5, 5, 6)

    summary(model=model, input_data=x, mode='train',
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            # row_settings=["var_names"]
            )
