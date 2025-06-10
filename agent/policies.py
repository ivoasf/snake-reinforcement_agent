"""
    An implementation of policies for reinforcement learning agents.
"""

import math
import torch
import random
import torch.nn.functional as F


class Policy:
    def choose_action(self, state, policy_net, device):
        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
    def __init__(self, num_actions: int = 3, eps_start: float = 0.9, eps_end: float = 0.1, eps_decay: int = 500):
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def choose_action(self, state, policy_net, device):
        sample = random.random()

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # use the policy network to get the action with the highest value
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long) 


class BoltzmannPolicy(Policy):
    def __init__(self, initial_temp: float = 1.0, min_temp: float = 0.1, temp_decay: float = 0.99):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.temperature = initial_temp
        self.steps_done = 0

    def choose_action(self, state, policy_net, device):
        with torch.no_grad():
            action_probs = F.softmax(policy_net(state) / self.temperature, dim=1)
            action = torch.multinomial(action_probs, num_samples=1).view(1, 1)

        self.steps_done += 1
        
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)

        return action
