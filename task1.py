"""
    A class representing the task1 agent for the snake game.

    The agent uses a simple DQN to play the snake game.
    The agent can choose actions based on a heuristic or random. 
"""

import cv2
import config
import time
import torch
import torch.nn as nn
from time import sleep
from itertools import count
from game.snake_game import SnakeGame
from game.game_wrapper import SnakeGameWrapper
from agent.dqn import SimpleDQN
from agent.heuristics import Heuristic, ImprovedHeuristic


class Task1:
    def __init__(
        self,
        device,
        optimizer,
        policy_net,
        heuristic: Heuristic,
        snake_game: SnakeGameWrapper,
    ):
        self.device = device
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.heuristic = heuristic
        self.snake_game = snake_game

    
    def choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).view(1, 1)


    def optimize_model(self, state, action, reward, next_state):
        state_action_values = self.policy_net(state).gather(1, action)

        if next_state is not None:
            max_next_q = self.policy_net(next_state).max(1)[0].detach()
            expected_state_action_values = reward + config.GAMMA * max_next_q
        else:
            expected_state_action_values = reward

        # compute the loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, use_heuristic=False, episodes=500, show_video=False, speed=0.001):
        print("\n-- Training --")

        if show_video:
            cv2.namedWindow("Training the agent!", cv2.WINDOW_NORMAL)

        scores = 0
        highest_score = 0

        for i_episode in range(episodes):
            pre_state, _, _, info = self.snake_game.reset()

            state = (
                torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                .permute(2, 3, 0, 1)
                .reshape(-1, config.SIZE[0], config.SIZE[1])
                .unsqueeze(0)
            )

            total_score = 0

            for t in count():
                if show_video:
                    cv2.imshow("Training the agent!", pre_state[:, :, :, -1])
                    cv2.waitKey(1) & 0xFF
                    sleep(speed)

                if use_heuristic:
                    action = torch.tensor([[self.heuristic.get_action(self.snake_game)]], device=self.device, dtype=torch.long)
                else:
                    action = self.choose_action(state)

                pre_state, reward, terminated, info = self.snake_game.step(action.item() - 1)

                if terminated:
                    next_state = None
                else:
                    next_state = (
                        torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                        .permute(2, 3, 0, 1)
                        .reshape(-1, config.SIZE[0], config.SIZE[1])
                        .unsqueeze(0)
                    )

                reward = torch.tensor([reward], device=self.device, dtype=torch.long)
                total_score = info["score"]
            
                self.optimize_model(state, action, reward, next_state)
                state = next_state

                if terminated:
                    break

            scores += total_score
            highest_score = max(highest_score, total_score)

            if (i_episode + 1) % 10 == 0:
                print(f"Episode {i_episode + 1} - Avg Score: {scores / 10}")
                scores = 0

        print(f"Highest score: {highest_score}")


    def test(self, episodes=10, show_video=True, speed=0.1):
        print("\n-- Testing --")

        scores = []

        if show_video:
            cv2.namedWindow("Agent playing the snake game!", cv2.WINDOW_NORMAL)

        max_score = 0

        for _ in range(episodes):
            pre_state, _, _, _ = self.snake_game.reset()

            for _ in count():
                if show_video:
                    cv2.imshow("Agent playing the snake game!", pre_state[:, :, :, -1])
                    cv2.waitKey(1) & 0xFF
                    sleep(speed)

                # (H, W, C, F)
                state_tensor = torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                # permute(2, 3, 0, 1) → (C, F, H, W)
                state_tensor = state_tensor.permute(2, 3, 0, 1)
                # reshape(1, -1, H, W) → (1, C × F, H, W)
                state_tensor = state_tensor.reshape(1, -1, state_tensor.shape[2], state_tensor.shape[3])

                action = self.choose_action(state_tensor)
                pre_state, _, terminated, info = self.snake_game.step(action.item() - 1)

                if terminated:
                    scores.append(info["score"])
                    max_score = max(max_score, info["score"])
                    break

        # one apple eaten is 1.0 point
        print(f"Average score over {episodes} episodes: {sum(scores) / len(scores):.2f}")
        print(f"Highest score: {max_score}")


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-- Device --")
    print(f"{device}")

    game = SnakeGame(config.WIDTH, config.HEIGHT, border=config.BORDER)
    snake_game = SnakeGameWrapper(game, num_frames=config.NUM_FRAMES)

    policy_net = SimpleDQN(config.INPUT_CHANNELS, config.NUM_ACTIONS).to(device)

    heuristic = ImprovedHeuristic()

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=config.LEARNING_RATE)

    agent = Task1(
        device=device,
        optimizer=optimizer,
        policy_net=policy_net,
        heuristic=heuristic,
        snake_game=snake_game
    )

    agent.train(use_heuristic=False, episodes=50, show_video=False, speed=0.0001)
    agent.test(episodes=10, show_video=False, speed=0.0001)

    end_time = time.time()
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        time_parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 or (hours == 0 and minutes == 0):
        time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    time_str = ", ".join(time_parts[:-1]) + (" and " if len(time_parts) > 1 else "") + time_parts[-1]
    print(f"\n-- Time --\n{time_str}")
