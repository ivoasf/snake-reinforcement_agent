""" 
    A Deep Q-Network (DQN) implementation for reinforcement learning in the snake game.
"""

import torch.nn as nn
import torch.nn.functional as F
from config import SIZE


class DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * SIZE[0] * SIZE[1], 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))  # (B, 32, H, W)
        x = F.relu(self.conv2(x))  # (B, 64, H, W)

        x = x.view(x.size(0), -1)  # flatten to (B, 64*H*W)
        x = F.relu(self.fc1(x))    # (B, 128)
        return self.fc2(x)         # (B, num_actions)

    
class SimpleDQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(SimpleDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(32 * SIZE[0] * SIZE[1], 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ComplexDQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(ComplexDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * SIZE[0] * SIZE[1], 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
