"""
    A class representing the task2 agent for the snake game.
"""

import cv2
import torch
import torch.nn as nn
import config
import time
from time import sleep
from itertools import count
from itertools import count
from game.snake_game import SnakeGame
from game.game_wrapper import SnakeGameWrapper
from agent.heuristics import Heuristic, MinDistanceHeuristic
from agent.dqn import SimpleDQN
from agent.replay_memory import ReplayMemory, Transition


class Task2:
    def __init__(
        self,
        device,
        optimizer,
        policy_net,
        target_net,
        heuristic: Heuristic,
        snake_game: SnakeGameWrapper,
        replay_memory=ReplayMemory(5000)
    ):
        self.device = device
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.target_net = target_net
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
                .reshape(-1, config.SIZE[0], config.SIZE[1])
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
                        .reshape(-1, config.SIZE[0], config.SIZE[1])
                        .unsqueeze(0)
                    )

                self.replay_memory.push(state, action, next_state, reward)

                state = next_state

                if done:
                    break


    def choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).view(1, 1)


    def optimize_model(self):
        if len(self.replay_memory) < config.BATCH_SIZE:
            return

        transitions = self.replay_memory.sample(config.BATCH_SIZE)
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
        next_state_values = torch.zeros(config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # compute the expected Q values
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

        # compute the loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train(self, use_heuristic=False, episodes=500, show_video=False, speed=0.001):
        print("\n-- Training --")

        if show_video:
            cv2.namedWindow("Training the agent!", cv2.WINDOW_NORMAL)

        scores = 0
        highest_score = 0

        for i_episode in range(episodes):
            update_target_every = 100  # hard update
            steps_done = 0

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

                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                total_score = info["score"]
            
                self.replay_memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                steps_done += 1
                if steps_done % update_target_every == 0:
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * config.TAU + target_net_state_dict[key] * (1 - config.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    break

            scores += total_score
            highest_score = max(highest_score, total_score)

            if i_episode % 10 == 0:
                print(f"Episode {i_episode} - Avg Score: {scores / 10}")
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
    target_net = SimpleDQN(config.INPUT_CHANNELS, config.NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    heuristic = MinDistanceHeuristic()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)

    agent = Task2(
        device=device,
        optimizer=optimizer,
        policy_net=policy_net,
        target_net=target_net,
        heuristic=heuristic,
        snake_game=snake_game
    )

    agent.train(use_heuristic=True, episodes=100, show_video=True, speed=0.0001)
    agent.test(episodes=10, show_video=True, speed=0.2)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
