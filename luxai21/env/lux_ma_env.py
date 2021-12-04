import copy
import datetime
import itertools
import json
import os.path
import pickle
import random
from functools import partial
from pathlib import Path
from typing import List, Mapping, Tuple, Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete, Dict, Box
from hydra import initialize, compose
from luxpythonenv.game.actions import MoveAction, SpawnCityAction, PillageAction, SpawnWorkerAction, SpawnCartAction, \
    ResearchAction
from luxpythonenv.game.constants import LuxMatchConfigs_Default
from omegaconf import open_dict
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from luxai21.env.render_utils import print_map
from luxai21.env.utils import *


class LuxMAEnv(MultiAgentEnv):
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

        if config["save_replay"]["wandb_every_x"] > 0:
            wandb.init(**config["wandb"])

        self.config = config

        self.game_config = config["game"]
        self.env_config = config["env"]

        # if seed is None use a random state from numpy
        # if seed -1 generate a random state in the beginning and use this for rendering
        self.seed = None
        self.randomize_seed = False

        lux_game_config = LuxMatchConfigs_Default.copy()
        lux_game_config.update(self.game_config)

        if "seed" in lux_game_config:
            if lux_game_config["seed"] == -1:
                self.randomize_seed = True
                self.seed = random.randint(0, 50000)
            else:
                self.seed = lux_game_config["seed"]

        self.game_state = Game(lux_game_config)
        self.game_state.skip_exception_logging = True

        # rendering
        # self.game_state.start_replay_logging(stateful=True)
        self.last_game_state: Optional[Any] = None
        self.last_game_stats: Optional[Any] = None  # to derive rewards per turn
        self.last_game_cities: Optional[Dict[str, City]] = None

        self.agents = []

        self.observation_space = self.worker_observation_space()

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

        # if true will log the individual rewards per unit and their amount
        self.reward_debug = False

        self.team_spirit = np.clip(config.get("team_spirit", 0.0), a_min=0.0, a_max=1.0)

        self.zero_sum = False

        self.reward_map = {
            # actions worker
            "move": 0,
            "transfer": 0,
            "build_city": 0.5,
            "pillage": 0,

            # actions citytile
            "build_worker": 1,
            "build_cart": 0.1,
            "research": 1,

            # agent
            # can also be acquired through transfer and lost in the night
            # this is already normalized such that 100 fuel are worth 1.
            "fuel_collected": 0.2,
            # TODO add discounted fuel_collected_at_night
            "fuel_dropped_at_city": 0.2,

            "death_before_end": 0,  # per turn away from 360

            # each turn
            "turn_unit": 0.1,
            "living_city_tiles": 0,  # get a reward for every living city_tile

            # all worker
            "death_city_tile": -0.1,

            # global
            "research_point": 0,
            "coal_researched": 0,
            "uranium_researched": 0,

            # end
            "win": 10,
            "end_city_tile": 1,
            # "premature_game_end": -0.1,  # per turn away from 360
        }

        self.action_reward_key = {
            # unit actions
            Constants.ACTIONS.MOVE: "move",
            Constants.ACTIONS.TRANSFER: "transfer",
            Constants.ACTIONS.BUILD_CITY: "build_city",
            Constants.ACTIONS.PILLAGE: "pillage",

            # city tile actions
            Constants.ACTIONS.BUILD_WORKER: "build_worker",
            Constants.ACTIONS.BUILD_CART: "build_cart",
            Constants.ACTIONS.RESEARCH: "research",
        }

        self._cumulative_rewards = dict()
        self.rewards = None
        self.dones = None
        self.infos = None

    def worker_observation_space(self):
        return gym.spaces.Dict(**{'map': Box(shape=(18, self.game_state.map.width, self.game_state.map.height),
                                             dtype=np.float64,
                                             low=-float('inf'),
                                             high=float('inf')
                                             ),
                                  'game_state': Box(shape=(2,),
                                                    dtype=np.float64,
                                                    low=float('-inf'),
                                                    high=float('inf')
                                                    ),
                                  'type': Discrete(3),
                                  'pos': Box(shape=(2,),
                                             dtype=np.float64,
                                             low=float('-inf'),
                                             high=float('inf')
                                             ),
                                  'action_mask': Box(shape=(12,),
                                                     dtype=np.float64,
                                                     low=0,
                                                     high=1
                                                     )})

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

    def render(self, mode='html', **kwargs) -> str:
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
            return ""

    def get_random_game_size(self) -> int:
        values = self.config["size_values"]
        probs = self.config["size_probs"]

        size = np.random.choice(values, p=probs)
        return size

    def reset(self) -> dict:
        """
        Returns:
            Observation of the first state
        """
        # get new map_seed
        if self.randomize_seed:
            seed = random.randint(0, 200000)
        else:
            seed = self.seed

        if self.config["random_game_size"]:
            size = self.get_random_game_size()
            self.game_state.configs["height"] = size
            self.game_state.configs["width"] = size

        self.game_state.configs["seed"] = seed
        self.game_state.reset()
        self.turn = 0

        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.last_game_state = pickle.loads(pickle.dumps(self.game_state.state))
        self.last_game_stats = {**self.game_state.stats}
        self.last_game_cities = pickle.loads(pickle.dumps(self.game_state.cities))

        obs = self.generate_obs()

        return obs

    def env_step(self, actions) -> Tuple[dict, dict, dict, dict, dict]:
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
            return {}, {}, {}, {}, {}

        for agent in actions:
            if isinstance(actions[agent], np.ndarray):
                actions[agent] = int(actions[agent][0])

        game_actions = self.translate_actions(actions)

        is_game_done = self.game_state.run_turn_with_actions(actions=game_actions)
        rewards, rewards_list, dones = self.compute_rewards()

        self.last_game_state = pickle.loads(pickle.dumps(self.game_state.state))
        self.last_game_cities = pickle.loads(pickle.dumps(self.game_state.cities))
        self.last_game_stats = {**self.game_state.stats}

        observations = self.generate_obs()

        infos = {piece_id: {} for piece_id in observations.keys()}

        # QUICK FIX TO RENDER EVERY GAME
        # TODO implement this functionality in logger
        if is_game_done:
            local_every_x = self.config["save_replay"].get("local_every_x", 0)
            if local_every_x > 0:
                if self.game_state.configs["seed"] % local_every_x == 0:
                    t = datetime.datetime.now().isoformat()
                    seed = self.game_state.configs["seed"]
                    if not os.path.exists("lux-replays"):
                        os.mkdir("lux-replays")
                    with open(f"lux-replays/{t}-{seed}.html", "w") as f:
                        f.write(self.render(mode="html"))

            wandb_every_x = self.config["save_replay"].get("wandb_every_x", 0)
            if wandb_every_x > 0:
                if self.game_state.configs["seed"] % wandb_every_x == 0:
                    wandb.log({"Replay": wandb.Html(self.render("html"), inject=False)})

        self.turn += 1

        return observations, rewards, dones, infos, rewards_list

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rewards, dones, infos, rewards_list = self.env_step(action_dict)
        return obs, rewards, dones, infos

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
            # agent_id = p0_u_1_4281
            team = int(agent_id[1])
            piece_id = agent_id[3:]
            if piece_id.startswith("ct"):
                ct_str, city_ident, city_id_nr, x_str, y_str, rand_id = piece_id.split("_")
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
                piece_id_parts = piece_id.split("_")
                piece_id = f"{piece_id_parts[0]}_{piece_id_parts[1]}"  # u_1
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

    def compute_rewards(self) -> Tuple[dict, dict, dict]:
        """
        Compute the rewards for all units for the current turn (given the last turn in self.last_game...)

        The values for the rewards are in self.reward_map, and the actions have to be resolved by self.action_reward_key
        action: 'm' -> key: 'move' -> reward: 0

        rewards are of the form ('id', value). This is that we can debug/log which rewards are given

        Returns:
            A dictionary with the reward for every unit (worker/cart/city_tile) that was either existed last turn,
            or this turn.

        """

        def total_fuel(u: Unit):
            return u.cargo["wood"] + u.cargo["coal"] * 10 + u.cargo["uranium"] * 40

        def fuel_score(x):
            """We map the fuel with a non-linear function such that
            f(100) ~= 1 (full cargo with wood)
            f(1000) ~= 2 (full cargo with coal)
            f(4000) ~= 3 (full cargo with uranium)
            """
            return (1.1 + np.exp(-x / 800)) * np.sqrt(x) / 21.1

        # TODO negative reward for units if city dies
        # TODO implement reward at turn end

        this_turn_units: Dict[str, Unit] = {**self.game_state.state["teamStates"][0]["units"],
                                            **self.game_state.state["teamStates"][1]["units"]}

        last_turn_units: Dict[str, Unit] = {**self.last_game_state["teamStates"][0]["units"],
                                            **self.last_game_state["teamStates"][1]["units"]}

        rewards_list = {}
        dones = {}

        # initialize all living units...
        for unit in this_turn_units.values():
            rewards_list[self.get_piece_id(unit.team, unit)] = [("", 0)]
            dones[self.get_piece_id(unit.team, unit)] = False

        # ...and city_tiles
        for city in self.game_state.cities.values():
            for cell in city.city_cells:
                city_tile = cell.city_tile
                rewards_list[self.get_piece_id(city.team, city_tile)] = [("", 0)]
                dones[self.get_piece_id(city.team, city_tile)] = False

        # loop over units of the last turn to find all death units
        for unit_id, unit in last_turn_units.items():
            if unit_id not in this_turn_units.keys():
                # that means unit died last turn
                value = np.minimum((1 - self.turn / 360) * self.reward_map["death_before_end"], 0)
                rewards_list[self.get_piece_id(unit.team, unit)] = [("death_before_end", value)]
                dones[self.get_piece_id(unit.team, unit)] = True

        is_game_over = self.game_state.match_over()
        dones["__all__"] = is_game_over
        winning_team = self.game_state.get_winning_team() if is_game_over else -1

        # units and city_tiles
        pieces = self.get_pieces()

        # ACTION
        # pieces that where created last turn start with a reward of zero
        for piece_id, piece in pieces.items():
            # Actions
            reward = ("no_action", 0)
            if piece.last_turn_action is not None:
                reward_key = self.action_reward_key[piece.last_turn_action.action]
                reward = (reward_key, self.reward_map[reward_key])

            rewards_list[piece_id].append(reward)

        # AGENT
        # loop over all living units (of both teams)
        for unit_id, unit in this_turn_units.items():
            team = unit.team
            piece_id = self.get_piece_id(team, unit)

            # check if it has collected new resources
            # check that the unit also existed last turn
            if unit.id in self.last_game_state["teamStates"][team]["units"].keys():
                unit_last_turn: Unit = self.last_game_state["teamStates"][team]["units"][unit.id]
                last_fuel_score = fuel_score(total_fuel(unit_last_turn))
                current_fuel_score = fuel_score(total_fuel(unit))

                # TODO does this catch the case when a unit is standing on the city adjacent to a resource tile?
                # check if unit stands on a city tile
                cell = self.game_state.map.get_cell_by_pos(unit.pos)
                if cell.city_tile and (cell.city_tile.team == team):
                    # on city, so it already dropped all its resources
                    assert current_fuel_score == 0
                    value = self.reward_map["fuel_dropped_at_city"] * fuel_score(getattr(unit, "fuel_deposited", 0))
                    rewards_list[piece_id].append(("fuel_dropped_at_city", value))
                    unit.fuel_deposited = 0
                else:
                    # not on city, forward difference in fuel score to reward (influenced by mining and the night)
                    fuel_difference = current_fuel_score - last_fuel_score
                    value = self.reward_map["fuel_collected"] * fuel_difference
                    rewards_list[piece_id].append(("fuel_collected", value))

            else:
                # can not collect resources if it was just created
                continue

        # compute how many citytiles per team
        count_city_tiles = {
            0: get_city_tile_count(self.game_state.cities, 0),
            1: get_city_tile_count(self.game_state.cities, 1)
        }

        last_count_city_tiles = {
            0: get_city_tile_count(self.last_game_cities, 0),
            1: get_city_tile_count(self.last_game_cities, 1),
        }

        # turn rewards and current city_tiles
        for unit_id, unit in this_turn_units.items():
            # turn reward
            piece_id = self.get_piece_id(unit.team, unit)
            rewards_list[piece_id].append(("turn_unit", self.reward_map["turn_unit"]))

            # reward for living city_tiles
            value = self.reward_map["living_city_tiles"] * count_city_tiles[unit.team]
            rewards_list[piece_id].append(("living_city_tiles", value))

        # penalize death of city_tiles
        current_city_ids = [city.id for city in self.game_state.cities.values()]
        lost_city_tiles = {
            0: 0,
            1: 0
        }
        for city in self.last_game_cities.values():
            if city.id not in current_city_ids:
                lost_city_tiles[city.team] += len(city.city_cells)

        for unit_id, unit in this_turn_units.items():
            team = unit.team
            piece_id = self.get_piece_id(team, unit)
            value = self.reward_map["death_city_tile"] * lost_city_tiles[team]
            rewards_list[piece_id].append(("death_city_tile", value))

        # global rewards
        # research & win
        for unit in this_turn_units.values():
            team = unit.team
            piece_id = self.get_piece_id(team, unit)
            # give research rewards only until uranium is researched, after that there is no point in researching
            if not self.last_game_state['teamStates'][team]['researched']['uranium']:
                value = self.reward_map["research_point"] * \
                        (self.game_state.state['teamStates'][team]['researchPoints'] -
                         self.last_game_state['teamStates'][team]['researchPoints'])
                rewards_list[piece_id].append(("research_point", value))

            # researched coal and uranium
            for resource in ['coal', 'uranium']:
                if self.game_state.state['teamStates'][team]['researched'][resource] and not \
                        self.last_game_state['teamStates'][team]['researched'][resource]:
                    rewards_list[piece_id].append((f"{resource}_researched", self.reward_map[f"{resource}_researched"]))

            # win
            if is_game_over:
                if team == winning_team:
                    value = (self.turn / 360) * self.reward_map["win"]
                    rewards_list[piece_id].append(("win", value))

                tiles = get_city_tile_count(self.game_state.cities, team)
                value = self.reward_map["end_city_tile"] * tiles
                rewards_list[piece_id].append(("end_city_tiles", value))

        # premature game end
        # if is_game_over:
        #     for unit_id, unit in {**this_turn_units, **last_turn_units}.items():
        #         piece_id = self.get_piece_id(unit.team, unit)
        #         value = (360 - self.turn) * self.reward_map["premature_game_end"]
        #         rewards_list[piece_id].append(("premature_game_end", value))

        rewards_sum = {}
        for piece_id, reward_list in rewards_list.items():
            total_reward = 0
            for ident, reward in reward_list:
                total_reward += reward
            rewards_sum[piece_id] = total_reward

        # apply team_spirit and zero sum
        total_team_reward = {
            0: 0,
            1: 0
        }
        team_size = {
            0: 0,
            1: 0
        }

        for piece_id, reward in rewards_sum.items():
            if "ct_" in piece_id:
                # skip city_tiles
                continue
            team = int(piece_id[1])
            team_size[team] += 1
            total_team_reward[team] += reward

        avg_team_reward = {
            0: total_team_reward[0] / team_size[0] if team_size[0] != 0 else 0,
            1: total_team_reward[1] / team_size[1] if team_size[1] != 0 else 0
        }

        # team spirit (this does not change the average over the team rewards)
        if self.team_spirit > 0:
            for piece_id, reward in rewards_sum.items():
                team = int(piece_id[1])
                rewards_sum[piece_id] = round(
                    (1 - self.team_spirit) * reward + self.team_spirit * avg_team_reward[team], 4)

        # zero sum
        if self.zero_sum:
            for piece_id, reward in rewards_sum.items():
                if "ct_" in piece_id:
                    # skip city_tiles
                    continue
                team = int(piece_id[1])
                other_team = (team + 1) % 2
                rewards_sum[piece_id] = reward - avg_team_reward[other_team]

        return rewards_sum, rewards_list, dones

    def get_pieces(self):
        pieces = {}

        for city in self.game_state.cities.values():
            for city_cell in city.city_cells:
                piece_id = self.get_piece_id(city.team, city_cell.city_tile)
                pieces[piece_id] = city_cell.city_tile

        for team in [Constants.TEAM.A, Constants.TEAM.B]:
            for unit in self.game_state.state["teamStates"][team]["units"].values():
                piece_id = self.get_piece_id(team, unit)
                pieces[piece_id] = unit

        return pieces

    def get_piece_id(self, team: int, piece: Union[CityTile, Unit]):
        seed = self.game_state.configs["seed"]
        if hasattr(piece, "cargo"):
            # is unit
            if seed is not None:
                return f"p{team}_{piece.id}_{seed}"
            else:
                return f"p{team}_{piece.id}_0"
        else:
            if seed is not None:
                return f"p{team}_ct_{piece.get_tile_id()}_{seed}"
            else:
                return f"p{team}_ct_{piece.get_tile_id()}_0"

    @staticmethod
    def piece_id_to_unit_id(piece_id):
        # p0_u_1_2348
        return "_".join(piece_id.split("_")[1:3])

    def _generate_unit_states(self, map_state: np.ndarray, team: int):
        """
        Return a dictionary where the keys are the unit_id or citytile_id and the value the unit
        {
            'u1': {
                type:  0=worker, 1=cart, 2=city_tile
                state: [cooldown, cargo], [fuel, fuel_burn, cooldown]
                pos: (x, y)
                action_mask: Discrete(12)

        }

        We have to pad the map because variable sizes maps will throw an error with rl lib
        """
        states = {}
        game_state_array = generate_simple_game_state_obs(self.game_state, team)

        for _, city in self.game_state.cities.items():
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    states[self.get_piece_id(team, city_tile)] = {
                        "pos": np.array([cell.pos.x, cell.pos.y]),
                        "action_mask": get_action_mask(self.game_state, team, None, city_tile, self.env_config),
                        "map": pad_map(append_position_layer(map_state, city_tile), 32),
                        "map_size": np.array([self.game_state.configs["height"]]),
                        # "mini_map": generate_mini_map(map_state, (cell.pos.x, cell.pos.y), config["fov"]),
                        "game_state": game_state_array
                    }

        for unit in self.game_state.state["teamStates"][team]["units"].values():
            states[self.get_piece_id(team, unit)] = {
                "pos": np.array([unit.pos.x, unit.pos.y]),
                "action_mask": get_action_mask(self.game_state, team, unit, None, self.env_config),
                "map": pad_map(append_position_layer(map_state, unit), 32),
                "map_size": np.array([self.game_state.configs["height"]]),
                "game_state": game_state_array
            }
        return states

    def generate_obs(self):
        _map_player0 = generate_simple_map_obs(self.game_state, 0)
        _map_player1 = generate_simple_map_obs(self.game_state, 1)

        unit_states_player0 = self._generate_unit_states(_map_player0, team=0)
        unit_states_player1 = self._generate_unit_states(_map_player1, team=1)

        return {
            **unit_states_player0,
            **unit_states_player1
        }

    @property
    def observation_spaces(self):
        return {
            piece_id: Dict({
                'map': Box(shape=(self.game_state.map.width, self.game_state.map.height, 10),
                           dtype=np.float32,
                           low=-float('inf'),
                           high=float('inf')
                           ),
                'game_state': Box(shape=(2,),
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
    initialize(config_path="../conf")
    config = compose(config_name="config")


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.shape
            return json.JSONEncoder.default(self, obj)


    with open_dict(config):
        config.env.env_config.game.seed = 123
        config.env.env_config.game.height = 8
        config.env.env_config.game.width = 8

    env = LuxMAEnv(config=config.env.env_config)
    obs = env.reset()

    p_id = None
    rewards = {}
    done = False
    while not done:
        actions = {}
        env.render(mode="cli")
        print(f"TURN: {env.turn}")

        # get the first worker of team 0
        for _piece_id, _ in obs.items():
            if _piece_id.startswith("p0_") and "ct" not in _piece_id:
                p_id = _piece_id
                action_id = input(f"enter action_id for {_piece_id}:")
                if action_id == "":
                    action_id = 0
                actions = {
                    _piece_id: int(action_id)
                }
                break
        map_obs = generate_simple_map_obs(game_state=env.game_state, team=0)
        game_obs = generate_simple_game_state_obs(game_state=env.game_state)

        obs, rewards, dones, infos = env.step(actions)
        done = env.game_state.match_over()

        if p_id in rewards:
            print(f"Reward: {rewards[p_id]}")
            print(f"Done: {dones[p_id]}")

    print("\nMatch over. End state:")
    env.render("cli")
