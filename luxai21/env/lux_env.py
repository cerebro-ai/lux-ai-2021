"""
Lux AI env following the DeepMind RL Environment API
https://github.com/deepmind/dm_env

"""
import copy
from typing import List, Optional

import numpy as np
from gym.spaces import Discrete, Dict, Box

from luxpythonenv.game.constants import LuxMatchConfigs_Default
from luxpythonenv.game.game import Game
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

from luxai21.env.utils import generate_map_state_matrix, switch_map_matrix_player_view, generate_game_state_matrix, \
    generate_unit_states

UNIT_FOV = 3


class LuxEnv(ParallelEnv):
    """
    Lux Multi Agent Environment following PettingZoo
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Dict where the two keys are respected:
                game: Config that gets passed to the game. Possible keys: width, height, seed
                agent: Config for the agent, e.g if agents should be able to build carts

        Example:
            {
                game: {
                    height: 12
                    width: 12
                    seed: 128343
                }
                agent: {
                    "allow_carts": False
                }
            }
        """
        super().__init__()  # does nothing

        game_config = config["game"]
        agent_config = config["agent"]

        self.game = Game(LuxMatchConfigs_Default.update(game_config))
        self.game_previous_turn: Optional[Game] = None  # to derive rewards per turn

        self.agent_config = {
            "allow_carts": False
        }.update(agent_config)

        self.agents = ["player_0", "player_1"]
        self.agent_name_mapping = {'player_0': 0, 'player_1': 1}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.steps = 0  # this is equivalent to turns in the game

        self.action_map = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            SpawnCityAction,
            PillageAction,
            None,  # City do nothing
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction
        ]

        self._cumulative_rewards = dict()
        self.rewards = None
        self.dones = None
        self.infos = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def state(self):
        pass

    def render(self, mode='human'):
        raise NotImplementedError()

    def reset(self):
        """
        TODO check return type
        """
        self.game.reset()
        self.steps = 0
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.game_previous_turn = copy.deepcopy(self.game)

        obs = self.generate_obs()

        return obs

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

        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        game_actions = self.translate_actions(actions)
        is_game_done = self.game.run_turn_with_actions(actions=game_actions)
        rewards = self.compute_rewards()

        self.game_previous_turn = copy.deepcopy(self.game)

        observations = self.generate_obs()

        infos = {agent: {} for agent in self.agents}
        dones = {agent: is_game_done for agent in self.agents}

        self.steps += 1

        return observations, rewards, dones, infos

    def translate_actions(self, actions) -> List:
        """
        TODO implement translate_actions
        """
        raise NotImplementedError

    def compute_rewards(self) -> dict:
        """
        TODO implement compute rewards
        """

        # check if game over
        if self.game.match_over():
            pass

        # return {
        #     agent: 0 for agent in self.agents
        # }

        raise NotImplementedError

    def generate_obs(self):
        _map_player0 = generate_map_state_matrix(self.game)
        _map_player1 = switch_map_matrix_player_view(_map_player0)

        game_state_player0 = generate_game_state_matrix(self.game, team=0)
        game_state_player1 = generate_game_state_matrix(self.game, team=1)

        unit_states_player0 = generate_unit_states(self.game, team=0, config=self.agent_config)
        unit_states_player1 = generate_unit_states(self.game, team=1, config=self.agent_config)

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

    @property
    def observation_spaces(self):
        return {self.agents[i]: Dict({
            '_map': Box(shape=(18, 32, 32),
                        dtype=np.float32,
                        low=-float('inf'),
                        high=float('inf')
                        ),
            '_game_state': Box(shape=(24,),
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
        return {agent: Discrete(12) for agent in self.agents}
