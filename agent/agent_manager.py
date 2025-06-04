"""
    A class for managing the saving and loading of agents.
"""

import os
import pickle
from copy import deepcopy
from agent.agent import Agent


class AgentManager:
    def __init__(self, agents_dir="agents"):
        self.agents_dir = agents_dir

    def save(self, agent: Agent, agent_id: str):
        print(f"\nSaving agent: {agent_id}")

        agents_dir = f"{self.agents_dir}{os.sep}{agent_id}"
        os.makedirs(agents_dir, exist_ok=True)
        agent_copy = deepcopy(agent)

        with open(f"{agents_dir}{os.sep}agent.pickle", "wb") as file:
            pickle.dump(agent_copy, file)

    def load(self, agent_id: str) -> Agent:
        print(f"Loading agent: {agent_id}")

        agents_dir = f"{self.agents_dir}{os.sep}{agent_id}"

        with open(f"{agents_dir}{os.sep}agent.pickle", "rb") as file:
            agent: Agent = pickle.load(file)

        return agent
