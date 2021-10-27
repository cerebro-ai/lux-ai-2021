from __future__ import annotations
from luxai21.agent.lux_agent import LuxAgent


class Stupid_Agent(LuxAgent):

    def __init__(self, learning: dict, model: dict, **kwargs):
        super(Stupid_Agent, self).__init__()

    def generate_actions(self, observation: dict):

        actions = {}
        return actions

    def receive_reward(self, reward: float, done: int):
        pass
