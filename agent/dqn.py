""" 
    A Deep Q-Network (DQN) implementation for reinforcement learning in the snake game.
"""

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 512) # assuming board size is 16x16
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))  # (B, 32, H, W)
        x = F.relu(self.conv2(x))  # (B, 64, H, W)
        x = F.relu(self.conv3(x))  # (B, 64, H, W)

        x = x.view(x.size(0), -1)  # flatten to (B, 64*H*W)
        x = F.relu(self.fc1(x))    # (B, 512)
        x = self.fc2(x)            # (B, num_actions)

        return x
