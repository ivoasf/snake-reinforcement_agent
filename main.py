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
from agent.agent_manager import AgentManager
from game.snake_game import SnakeGame
from game.game_wrapper import SnakeGameWrapper


def train_agent(device, agent: Agent, episodes=500, show_video=False, speed=0.1):
    print("\n-- Training --")

    if show_video:
        cv2.namedWindow("Training the agent!", cv2.WINDOW_NORMAL)

    scores = 0
    highest_score = 0

    for i_episode in range(episodes):
        pre_state, _, _, info = agent.snake_game.reset()

        state = (
            torch.tensor(pre_state, dtype=torch.float32, device=device)
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

            action = agent.choose_action(state)
            pre_state, reward, terminated, info = agent.snake_game.step(action.item() - 1)
        
            reward = torch.tensor([reward], device=device)
            total_score = info["score"]

            if terminated:
                next_state = None
            else:
                next_state = (
                    torch.tensor(pre_state, dtype=torch.float32, device=device)
                    .permute(2, 3, 0, 1)
                    .reshape(-1, config.SIZE[0], config.SIZE[1])
                    .unsqueeze(0)
                )
        
            agent.replay_memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()

            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * config.TAU + target_net_state_dict[key] * (1 - config.TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if terminated:
                break

        scores += total_score
        highest_score = max(highest_score, total_score)

        if i_episode % 5 == 0:
            print(f"Episode {i_episode} - Avg Score: {scores / 5}")
            scores = 0

    print(f"Highest score: {highest_score}")


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
    print(f"{device}\n")

    agent_manager = AgentManager()
    agent_id = "1st"

    try:
        agent = agent_manager.load(agent_id)
        print("[✓] Agent loaded successfully!")
    except FileNotFoundError:
        print("[!] No saved agent found, creating a new one...")

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

    train_agent(device, agent, episodes=100, show_video=False, speed=0.001)
    agent_manager.save(agent, agent_id)

    test_agent(device, agent, episodes=11, show_video=True, speed=0.2)


if __name__ == "__main__":
    main()
