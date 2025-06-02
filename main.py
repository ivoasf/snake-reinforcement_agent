"""
    Main entry point for the snake reinforcement learning agent.
"""

import cv2
import torch
import config
from time import sleep
from itertools import count
from agent.agent import Agent
from agent.dqn import DQN
from agent.heuristics import MinDistanceHeuristic
from agent.policies import EpsilonGreedyPolicy
from game.snake_game import SnakeGame
from game.game_wrapper import SnakeGameWrapper


def train_agent(device, show_video=False) -> Agent:
    game = SnakeGame(config.WIDTH, config.HEIGHT, border=config.BORDER)
    snake_game = SnakeGameWrapper(game, num_frames=config.NUM_FRAMES)

    policy = EpsilonGreedyPolicy()
    heuristic = MinDistanceHeuristic()

    policy_net = DQN(config.INPUT_CHANNELS, config.NUM_ACTIONS).to(device)
    target_net = DQN(config.INPUT_CHANNELS, config.NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)

    agent = Agent(
        device=device,
        optimizer=optimizer,
        policy_net=policy_net,
        target_net=target_net,
        policy=policy,
        heuristic=heuristic,
        snake_game=snake_game
    )

    agent.train(num_episodes=31, show_video=show_video)

    return agent


def test_agent(device, agent: Agent, episodes=10, show_video=True, speed=0.2):
    print("\n-- Testing --")

    scores = []

    if show_video:
        cv2.namedWindow("Agent playing the snake game!", cv2.WINDOW_NORMAL)

    max_score = 0

    for _ in range(episodes):
        pre_state, _, _, _ = agent.snake_game.reset()

        for _ in count():
            if show_video:
                cv2.imshow("Agent playing the snake game!", pre_state[:, :, :, -1])
                cv2.waitKey(1) & 0xFF
                sleep(speed)

            # (H, W, C, F)
            state_tensor = torch.tensor(pre_state, dtype=torch.float32, device=device)
            # permute(2, 3, 0, 1) → (C, F, H, W)
            state_tensor = state_tensor.permute(2, 3, 0, 1)
            # reshape(1, -1, H, W) → (1, C × F, H, W)
            state_tensor = state_tensor.reshape(1, -1, state_tensor.shape[2], state_tensor.shape[3])

            action = agent.choose_action(state_tensor)
            pre_state, _, terminated, info = agent.snake_game.step(action.item() - 1)

            if terminated:
                scores.append(info["score"])
                max_score = max(max_score, info["score"])
                break

    # one apple eaten is 1.0 point
    print(f"Average score over {episodes} episodes: {sum(scores) / len(scores):.2f}")
    print(f"Highest score: {max_score}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-- Device --")
    print(f"{device}")

    agent = train_agent(device, show_video=True)
    test_agent(device, agent, episodes=10, show_video=True, speed=0.4)


if __name__ == "__main__":
    main()
