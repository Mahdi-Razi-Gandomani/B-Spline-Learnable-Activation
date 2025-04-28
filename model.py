import torch.nn as nn
import torch.nn.functional as F
from bspline_activation import LearnableBSplineActivation


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learnable=False):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = LearnableBSplineActivation(start_point=-2, end_point=2) if learnable else F.relu

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class BaseRegression(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LearnableRegression(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.activation1 = LearnableBSplineActivation(start_point=-1, end_point=1)
        self.activation2 = LearnableBSplineActivation(start_point=-1, end_point=1)

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.fc3(x)
        return x
