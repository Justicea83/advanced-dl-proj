import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, Adagrad


class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MusicGenreClassifier, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 512)  # First hidden layer
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer
        self.fc3 = nn.Linear(256, 128)  # Third hidden layer
        self.fc4 = nn.Linear(128, output_size)  # Output layer

        # Initialize weights using Glorot initialization and biases to zero
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_adam_optimizer(self):
        return Adam(self.parameters())

    def get_rms_optimizer(self):
        return RMSprop(self.parameters())

    def get_ada_optimizer(self):
        return Adagrad(self.parameters())
