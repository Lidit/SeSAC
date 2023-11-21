import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=2)
        self.act1 = nn.Sigmoid()

        self.fc2 = nn.Linear(in_features=2, out_features=1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        pred = self.act2(x)
        pred = pred.view(-1)
        return pred