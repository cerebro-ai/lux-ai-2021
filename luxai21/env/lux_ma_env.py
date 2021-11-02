import json
import random
from functools import partial
from pathlib import Path

import numpy as np
from typing import List, Mapping, Tuple, Any, Union
import copy
import wandb
from gym.spaces import Discrete, Dict, Box
from luxpythonenv.game.actions import MoveAction, SpawnCityAction, PillageAction, SpawnWorkerAction, SpawnCartAction, \
    ResearchAction, Action
from luxpythonenv.game.constants import LuxMatchConfigs_Default
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

from luxai21.env.render_utils import print_map
from luxai21.env.utils import *


class LuxMAEnv(ParallelEnv):
    """
    Lux Multi Agent Environment following PettingZoo
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Dict where the two keys are respected:
                game: Config that gets passed to the game. Possible keys: width, height, seed
                env: Config for the agent, e.g if agents should be able to build carts
                reward: Config which describes the value of every reward

        Example:
            {
                game: {
                    height: 12
                    width: 12
                    seed: 128343
                }
                env: {
                    "allow_carts": False
                }
                reward: {
                    "BUILD_CITY_TILE": 0.01
                    ...
                }
            }
        """
        super().__init__()  # does nothing

        self.game_config = config["game"]
        self.env_config = config["env"]
        self.seed = None

        lux_game_config = LuxMatchConfigs_Default.copy()
        lux_game_config.update(self.game_config)
        if "seed" not in lux_game_config or lux_game_config["seed"] is None:
            lux_game_config["seed"] = random.randint(0, 50000)
        else:
            self.seed = lux_game_config["seed"]

        self.game_state = Game(lux_game_config)
        # rendering
        self.game_state.start_replay_logging(stateful=True)
        self.last_game_state: Optional[Any] = None  # to derive rewards per turn
        self.last_game_cities: Optional[Dict] = None

        self.env_config = {
            "allow_carts": False
        }
        self.env_config.update(self.env_config)

        self.agents = []

        self.turn = 0

        self.unit_action_map = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),  # 1
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),  # 2
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),  # 3
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),  # 4
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            SpawnCityAction,  # 7
            PillageAction,  # 8
        ]

        self.city_tile_action_map = [
            None,  # City do nothing
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction
        ]

        # TODO take reward from config
        self.reward_map = {
            Constants.ACTIONS.MOVE: 0,
            Constants.ACTIONS.TRANSFER: 0,
            Constants.ACTIONS.BUILD_CITY: 1,
            Constants.ACTIONS.PILLAGE: 0,
            # city tiles
            Constants.ACTIONS.BUILD_WORKER: 1,
            Constants.ACTIONS.BUILD_CART: 1,
            Constants.ACTIONS.RESEARCH: 1
        }

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

    def _get_replay_steps(self):
        steps = []
        commands = self.game_state.replay_data["allCommands"]
        for turn in commands:
            turn_steps = []
            for cmd in turn:
                step = {
                    "action": [cmd["command"]],
                    "observation": {
                        "player": cmd["agentID"]
                    }
                }
                turn_steps.append(step)

            steps.append(turn_steps)
        return steps

    def _render_env(self):
        # spec = json.load(Path(__file__).parent.joinpath("specification.json").open())
        result = {
            "steps": self._get_replay_steps(),
            # "allCommands": self.game_state.replay_data["allCommands"],
            "mapType": Constants.MAP_TYPES.RANDOM,
            "configuration": {
                **self.game_config,
                "seed": self.game_state.replay_data["seed"],
            },
            "info": {
                "TeamNames": self.agents
            },
            "teamDetails": [
                {
                    "name": "Agent0",
                    "tournamentID": '',
                },
                {
                    "name": "Agent1",
                    "tournamentID": '',
                },
            ],
            "version": "3.1.0"
        }
        return result

    def render(self, mode='html', **kwargs):
        def get_html(input_html):
            key = "/*window.kaggle*/"
            value = f"""window.kaggle = {json.dumps(input_html, indent=2)};\n\n"""

            with Path(__file__).parent.joinpath("render_index.html").open("r", encoding="utf-8") as f:
                result = f.read()
                result = result.replace(key, value)
            return result

        if mode == "html" or mode == "ipython":
            # is_playing = not self.game_state.match_over()
            window_kaggle = {
                "debug": self.game_state.configs["debug"],
                "playing": True,
                "step": 0,
                "controls": True,
                "environment": self._render_env(),
                "logs": "",
                **kwargs,
            }

            player_html = get_html(window_kaggle)

            if mode == "html":
                return player_html
            elif mode == "ipython":
                # from IPython.display import display, HTML
                player_html = player_html.replace('"', '&quot;')
                width = 300
                height = 300
                html = f'<iframe srcdoc="{player_html}" width="{width}" height="{height}" frameborder="0"></iframe> '
                return html

        elif mode == "cli":
            print_map(self.game_state.map)

    def reset(self):
        """
        Returns:
            Observation of the first state
        """
        # get new map_seed
        if self.seed is None:
            seed = random.randint(0, 200000)
        else:
            seed = self.seed

        self.game_state.configs["seed"] = seed
        self.game_state.reset()
        self.turn = 0

        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.last_game_state = copy.deepcopy(self.game_state.state)
        self.last_game_cities = copy.deepcopy(self.game_state.cities)

        obs = self.generate_obs()

        return obs

    def step(self, actions) -> Tuple[dict, dict, dict, dict]:
        """
        Args:
            actions: Dict
                {
                'p0_u1': 7,
                'p0_ct_2_2': 2
                'p1_u3': 4,
                'p1_ct_4_4': 1,
                }
        Returns:
            observations, rewards, dones, infos
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        game_actions = self.translate_actions(actions)
        assert len(actions.keys()) == len(game_actions)

        is_game_done = self.game_state.run_turn_with_actions(actions=game_actions)
        rewards = self.compute_rewards()

        self.last_game_state = copy.deepcopy(self.game_state.state)
        self.last_game_cities = copy.deepcopy(self.game_state.cities)

        observations = self.generate_obs()

        infos = {piece_id: {} for piece_id in observations.keys()}

        """
        For every agent for which we received an action,
        but it is not any more in observation set done to true
        """
        dones = {piece_id: False for piece_id in observations.keys()}
        for piece_id in actions.keys():
            if piece_id not in dones:
                dones[piece_id] = True
            if piece_id not in rewards:
                # last reward
                rewards[piece_id] = 0

        dones["__all__"] = is_game_done

        self.turn += 1

        return observations, rewards, dones, infos

    def translate_actions(self, actions: Mapping[str, int]) -> List:
        """
        Args:
            actions: Dict with the actions of both agents/players for each of their units/city_tiles
                e.g. {
                'player_0': {
                    'u1': 1,
                    ...
                },
                'player_1': {
                    'c2': 3,
                    ...
                }

        Returns:
            A list with all of the actions from both players
        """
        translated_actions = []

        # agent: p0_
        for agent_id, action_id in actions.items():
            team = int(agent_id[1])
            piece_id = agent_id[3:]
            if piece_id.startswith("ct"):
                ident, x_str, y_str = piece_id.split("_")
                cell = self.game_state.map.get_cell(int(x_str), int(y_str))
                city_tile = cell.city_tile
                if city_tile is None:
                    raise Exception(f"city_tile could not be found for {piece_id}")

                action_class = self.city_tile_action_map[action_id]
                if action_class is not None:
                    action = action_class(
                        game=self.game_state,
                        unit_id=None,
                        unit=None,
                        city_id=city_tile.city_id,
                        city_tile=city_tile,
                        team=team,
                        x=int(x_str),
                        y=int(y_str)
                    )

                    translated_actions.append(action)

            else:
                unit = self.game_state.get_unit(team=team, unit_id=piece_id)
                action = self.unit_action_map[action_id](
                    game=self.game_state,
                    unit_id=unit.id,
                    unit=unit,
                    city_id=None,
                    citytile=None,
                    team=team,
                    x=unit.pos.x,
                    y=unit.pos.y
                )
                translated_actions.append(action)

        return translated_actions

    def compute_rewards(self) -> dict:

        # TODO negative reward for units if city dies

        rewards = {}
        for piece_id, piece in self.get_pieces().items():
            if piece.last_turn_action is not None:
                reward = self.reward_map[piece.last_turn_action.action]
                rewards[piece_id] = reward
            else:
                rewards[piece_id] = 0
        return rewards

    def get_pieces(self):

        pieces = {}

        for city in self.game_state.cities.values():
            for city_cell in city.city_cells:
                piece_id = get_piece_id(city.team, city_cell.city_tile)
                pieces[piece_id] = city_cell.city_tile

        for team in [Constants.TEAM.A, Constants.TEAM.B]:
            for unit in self.game_state.state["teamStates"][team]["units"].values():
                piece_id = get_piece_id(team, unit)
                pieces[piece_id] = unit

        return pieces

    def generate_obs(self):

        _map_player0 = generate_map_state_matrix(self.game_state)
        _map_player1 = switch_map_matrix_player_view(_map_player0)

        unit_states_player0 = generate_unit_states(self.game_state, _map_player0, team=0, config=self.env_config)
        unit_states_player1 = generate_unit_states(self.game_state, _map_player1, team=1, config=self.env_config)

        return {
            **unit_states_player0,
            **unit_states_player1
        }

    @property
    def observation_spaces(self):
        return {
            piece_id: Dict({
                'map': Box(shape=(18, self.game_state.map.width, self.game_state.map.height),
                           dtype=np.float32,
                           low=-float('inf'),
                           high=float('inf')
                           ),
                'game_state': Box(shape=(24,),
                                  dtype=np.float32,
                                  low=float('-inf'),
                                  high=float('inf')
                                  ),
                'type': Discrete(3),
                'pos': Box(shape=(2,),
                           dtype=np.int32,
                           low=float('-inf'),
                           high=float('inf')
                           ),
                'action_mask': Box(shape=(12,),
                                   dtype=np.int,
                                   low=0,
                                   high=1
                                   ),
            }) for piece_id, piece in self.get_pieces().items()}

    @property
    def action_spaces(self):
        return {piece_id: Discrete(len(self.unit_action_map)) if isinstance(piece, Unit) else Discrete(
            len(self.city_tile_action_map)) for
                piece_id, piece in self.get_pieces().items()}


if __name__ == '__main__':
    config = {
        "game": {
            "height": 12,
            "width": 12,
            "seed": 21
        },
        "env": {
            "allow_carts": False
        }

    }


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.shape
            return json.JSONEncoder.default(self, obj)


    env = LuxMAEnv(config=config)
    obs = env.reset()

    p_id = None
    rewards = {}
    done = False
    while not done:
        actions = {}
        env.render(mode="cli")
        print(f"TURN: {env.turn}")

        # get the first worker of team 0
        for piece_id, piece in obs.items():
            if piece_id.startswith("p0_") and "ct" not in piece_id:
                p_id = piece_id
                action_id = input(f"enter action_id for {piece_id}:")
                if action_id == "":
                    action_id = 0
                actions = {
                    piece_id: int(action_id)
                }
                break

        obs, rewards, dones, infos = env.step(actions)
        done = env.game_state.match_over()

        if p_id in rewards:
            print(f"Reward: {rewards[p_id]}")
            print(f"Done: {dones[p_id]}")

    print("\nMatch over. End state:")
    env.render("cli")
