"""
Lux AI env following the DeepMind RL Environment API
https://github.com/deepmind/dm_env

"""
import dm_env
import numpy as np
from gym.spaces import Discrete, Dict, Box

from luxpythonenv.game.constants import LuxMatchConfigs_Default
from luxpythonenv.game.game import Game
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

UNIT_FOV = 3


class LuxEnv(AECEnv):
    """
    Lux Multi Agent Environment following PettingZoo
    """

    def __init__(self, unit_fov: int = 3, game_config=None):
        """
        Args:
            unit_fov: Field of View, how far can a unit (worker, cart) see in each direction
            game_config: Config that gets passed to the game. Possible keys:
                width, height, seed
        """
        super().__init__()  # does nothing

        self.game_config = LuxMatchConfigs_Default
        if game_config is not None:
            self.game_config.update(game_config)

        self.unit_fov = unit_fov
        self.game = Game(configs=game_config)

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {'player_0': 0, 'player_1': 1}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.observation_spaces = self.observation_spec()
        self.action_spaces = self.observation_spec()

        self.steps = 0  # this is equivalent to turns in the game

        self.rewards = None
        self.dones = None
        self.infos = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self) -> dm_env.TimeStep[float, float, dict]:
        self.game.reset(increment_turn=False)
        self.steps = 0
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        obs = self.get_observations(self.game)

        return None

    def get_observations(self, game):
        """
        return {
            'player1': {
                '_map_': map_obs,
                'u1': unit_obs
            },
            'player2': {

        }
        """

        # create an observation for each team

        return {
            self.agents[0],
        }

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        agent = self.agent_selection
        action = np.array(action, dtype=np.float32)
        is_last = self._agent_selector.is_last()

        self.env.step(action, self.agent_name_mapping[agent], is_last)

        if is_last:
            last_rewards = self.env.get_last_rewards()
            for r in self.rewards:
                self.rewards[r] = last_rewards[self.agent_name_mapping[r]]
            for d in self.dones:
                self.dones[d] = self.env.get_last_dones()[self.agent_name_mapping[d]]
            self.agent_name_mapping = {agent: i for i, (agent, done) in
                                       enumerate(zip(self.possible_agents, self.env.get_last_dones()))}
            iter_agents = self.agents[:]
            for a, d in self.dones.items():
                if d:
                    iter_agents.remove(a)
            self._agent_selector.reinit(iter_agents)
        else:
            self._clear_rewards()
        if self._agent_selector.agent_order:
            self.agent_selection = self._agent_selector.next()

        if self.env.frames >= self.env.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        self._dones_step_first()
        self.steps += 1

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def render(self, mode='human'):
        raise NotImplementedError()

    def state(self):
        pass

    @property
    def observation_spaces(self):
        return {agent: Dict({
            'map_obs': Box(shape=(18, 32, 32),
                           dtype=np.float32,
                           low=-float('inf'),
                           high=float('inf')
                           ),
            'game_stats': Box(shape=(22,),
                              dtype=np.float32,
                              low=float('-inf'),
                              high=float('inf')
                              ),
            'unit_type': Discrete(3),
            'unit_obs': Box(shape=(18, 2 * UNIT_FOV + 1, 2 * UNIT_FOV + 1),
                            dtype=np.float32,
                            low=float('-inf'),
                            high=float('inf')
                            ),
            'unit_stats': Box(shape=(3,),
                              dtype=np.float32,
                              low=float('-inf'),
                              high=float('inf')
                              ),
            'action_mask': Box(shape=(12,),
                               dtype=np.int,
                               low=0,
                               high=1
                               ),
        }) for agent in self.possible_agents}

    @property
    def actions_spaces(self):
        return {agent: Discrete(12) for agent in self.possible_agents}
