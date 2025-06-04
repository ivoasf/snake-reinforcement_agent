"""
    A class representing the agent that interacts with the snake game environment.
"""

import torch
import torch.nn as nn
from config import SIZE, BATCH_SIZE, GAMMA
from itertools import count
from agent.policies import Policy
from agent.heuristics import Heuristic
from agent.replay_memory import ReplayMemory, Transition
from game.game_wrapper import SnakeGameWrapper


class Agent:
    def __init__(
        self,
        device,
        optimizer,
        policy_net,
        target_net,
        policy: Policy,
        heuristic: Heuristic,
        snake_game: SnakeGameWrapper,
        replay_memory=ReplayMemory(10000)
    ):
        self.device = device
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.target_net = target_net
        self.policy = policy
        self.heuristic = heuristic
        self.snake_game = snake_game
        self.replay_memory = replay_memory

        self.build_memory(self.snake_game)


    def build_memory(self, game: SnakeGameWrapper):
        while len(self.replay_memory) < self.replay_memory.memory.maxlen:
            pre_state, _, _, _ = game.reset()

            state = (
                torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                .permute(2, 3, 0, 1)
                .reshape(-1, SIZE[0], SIZE[1])
                .unsqueeze(0)
            )

            for j in count():
                action = self.heuristic.get_action(game)
                pre_state, reward, done, _ = game.step(action - 1)

                reward = torch.tensor([reward], device=self.device, dtype=torch.long)
                action = torch.tensor([[action]], device=self.device, dtype=torch.long)

                if done:
                    next_state = None
                else:
                    next_state = (
                        torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                        .permute(2, 3, 0, 1)
                        .reshape(-1, SIZE[0], SIZE[1])
                        .unsqueeze(0)
                    )

                self.replay_memory.push(state, action, next_state, reward)

                state = next_state

                if done:
                    break


    def choose_action(self, state):
        return self.policy.choose_action(state, self.policy_net, self.device)


    def optimize_model(self):
        if len(self.replay_memory) < BATCH_SIZE:
            return
        
        transitions = self.replay_memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, a) where a is the action taken by the agent
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # compute V(s_{t+1}) for the next states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # compute the loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
