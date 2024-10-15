import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=5, stride=2)  # Output: [N, 16, 40, 40]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)        # Output: [N, 32, 18, 18]
        self.in_features = 32 * 18 * 18
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(-1, self.in_features)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
