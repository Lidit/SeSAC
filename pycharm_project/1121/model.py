import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=10)
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(in_features=10, out_features=20)
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(in_features=20, out_features=4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)

        x = self.fc3(x)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=392)
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(in_features=392, out_features=196)
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)

        x = self.fc3(x)
        return x
