"""
Lux AI env following the DeepMind RL Environment API
https://github.com/deepmind/dm_env

"""
import copy

import dm_env
import numpy as np
from gym.spaces import Discrete, Dict, Box
from kaggle_environments import make

from luxpythonenv.game.constants import LuxMatchConfigs_Default
from luxpythonenv.game.game import Game
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector

UNIT_FOV = 3


class LuxEnv(ParallelEnv):
    """
    Lux Multi Agent Environment following PettingZoo
    """

    def __init__(self, game_config: dict = None):
        """
        Args:
            unit_fov: Field of View, how far can a unit (worker, cart) see in each direction
            game_config: Config that gets passed to the game. Possible keys:
                width, height, seed
        """
        super().__init__()  # does nothing

        self.game = Game(LuxMatchConfigs_Default.update(game_config))
        self.previous_turn_game = None  # to derive rewards per turn

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {'player_0': 0, 'player_1': 1}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.steps = 0  # this is equivalent to turns in the game

        self._cumulative_rewards = dict()
        self.rewards = None
        self.dones = None
        self.infos = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self) -> dm_env.TimeStep[float, float, dict]:
        self.game.reset()
        self.steps = 0
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.previous_turn_game = copy.deepcopy(self.game)

        # obs = self.get_observations(self.game)

        return None

    def step(self, actions):
        """
        Args:
            actions: Dict
                {
                'player0': {
                    'u1': 7,
                    'c2': 2
                    },
                'player1': {
                    'u3': 4,
                    'c2': 1,
                    }
                }

        TODO implement translate_actions
        TODO implement compute_rewards
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        game_actions = translate_actions(actions)
        is_game_done = self.game.run_turn_with_actions(actions=game_actions)
        rewards = compute_rewards(self.previous_turn_game, self.game)
        observations = self.generate_obs()

        infos = {agent: {} for agent in self.agents}
        dones = {agent: is_game_done for agent in self.agents}

        self.steps += 1

        return observations, rewards, dones, infos

    def generate_obs(self):
        """
        TODO generate_map_state_matrix
        TODO switch_player_view
        TODO generate_game_state_matrix
        TODO generate_unit_states
        """

        _map_player0 = generate_map_state_matrix(self.game)
        _map_player1 = switch_player_view(_map_player0)

        game_state_player0 = generate_game_state_matrix(self.game, 0)
        game_state_player1 = generate_game_state_matrix(self.game, 1)

        unit_states_player0 = generate_unit_states(game=self.game, team=0)
        unit_states_player1 = generate_unit_states(game=self.game, team=1)

        return {
            self.agents[0]: {
                '_map': _map_player0,
                '_game_state': game_state_player0,
                **unit_states_player0
            },
            self.agents[1]: {
                '_map': _map_player1,
                '_game_state': game_state_player1,
                **unit_states_player1
            }
        }

    def render(self, mode='human'):
        raise NotImplementedError()

    @property
    def observation_spaces(self):
        return {self.agents[i]: Dict({
            '_map': Box(shape=(18, 32, 32),
                        dtype=np.float32,
                        low=-float('inf'),
                        high=float('inf')
                        ),
            '_game_state': Box(shape=(22,),
                               dtype=np.float32,
                               low=float('-inf'),
                               high=float('inf')
                               ),
            **{
                unit: Dict({
                    'type': Discrete(3),
                    'state': Box(shape=(3,),
                                 dtype=np.float32,
                                 low=float('-inf'),
                                 high=float('inf')
                                 ),
                    'action_mask': Box(shape=(12,),
                                       dtype=np.int,
                                       low=0,
                                       high=1
                                       ),
                })
                for unit in self.game.get_teams_units(i)
            }
        }) for i in [0, 1]}

    @property
    def action_spaces(self):
        return {agent: Discrete(12) for agent in self.possible_agents}
