import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 64, kernel_size=8, stride=4)  # Output: [N, 64, 20, 20]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)       # Output: [N, 128, 9, 9]
        self.in_features = 128 * 9 * 9
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
