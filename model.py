import torch.nn as nn
from learnableSplineActivation import LearnableBSplineActivation

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 learnable_spline_activation=False, num_control_points=5, degree=3):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.activation = LearnableBSplineActivation(num_control_points, degree) if learnable_spline_activation is True else nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x