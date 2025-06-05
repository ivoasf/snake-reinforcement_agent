"""
    Main entry point for the snake reinforcement learning agent.
"""

import torch
import config
from agent.agent import Agent
from agent.dqn import DQN
from agent.heuristics import MinDistanceHeuristic
from agent.policies import EpsilonGreedyPolicy
from agent.agent_manager import AgentManager
from game.snake_game import SnakeGame
from game.game_wrapper import SnakeGameWrapper


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

    agent.train(episodes=21, show_video=True, speed=0.001)
    agent_manager.save(agent, agent_id)

    agent.test(episodes=11, show_video=True, speed=0.2)


if __name__ == "__main__":
    main()
