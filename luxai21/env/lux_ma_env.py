import copy
import datetime
import json
import os.path
import random
from functools import partial
from pathlib import Path
from typing import List, Mapping, Tuple, Any

import gym.spaces
from gym.spaces import Discrete, Dict, Box
from hydra import initialize, compose
from luxpythonenv.game.actions import MoveAction, SpawnCityAction, PillageAction, SpawnWorkerAction, SpawnCartAction, \
    ResearchAction
from luxpythonenv.game.constants import LuxMatchConfigs_Default
from omegaconf import open_dict
from ray.rllib import MultiAgentEnv

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
        self.last_game_state: Optional[Any] = None
        self.last_game_stats: Optional[Any] = None  # to derive rewards per turn
        self.last_game_cities: Optional[Dict] = None

        self.env_config = {
            "allow_carts": False
        }
        self.env_config.update(self.env_config)

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

        self.team_spirit = np.clip(config.get("team_spirit", 0.0), a_min=0.0, a_max=1.0)
        self.reward_map = dict(config["reward"])

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
                                  'game_state': Box(shape=(24,),
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
        self.last_game_stats = copy.deepcopy(self.game_state.stats)
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

        for agent in actions:
            if isinstance(actions[agent], np.ndarray):
                actions[agent] = int(actions[agent][0])

        game_actions = self.translate_actions(actions)

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
        dones = {piece_id: is_game_done for piece_id in observations.keys()}
        for piece_id in actions.keys():
            if piece_id not in dones:
                dones[piece_id] = True
            if piece_id not in rewards:
                # last reward
                rewards[piece_id] = 0

        dones["__all__"] = is_game_done

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

    def compute_rewards(self) -> dict:

        # TODO negative reward for units if city dies
        # TODO implement reward at turn end

        rewards = {}
        is_game_over = self.game_state.match_over()
        winning_team = self.game_state.get_winning_team()

        pieces = self.get_pieces()

        for piece_id, piece in pieces.items():
            reward = 0
            # Actions
            if piece.last_turn_action is not None:
                reward = self.reward_map[self.action_reward_key[piece.last_turn_action.action]]
            rewards[piece_id] = reward

        current_city_ids = [city.id for city in self.game_state.cities.values()]

        lost_city = {
            0: False,
            1: False
        }

        for city in self.last_game_cities.values():
            if city.id not in current_city_ids:
                lost_city[city.team] = True

        if is_game_over:
            city_tiles_team1 = 0
            city_tiles_team2 = 0
            for _, city in self.game_state.cities.items():
                if city.team == 0:
                    city_tiles_team1 += len(city.city_cells)
                else:
                    city_tiles_team2 += len(city.city_cells)

        # Turn reward & death city
        for piece_id in rewards.keys():
            team = int(piece_id[1])

            if "ct_" in piece_id:
                rewards[piece_id] += self.reward_map["turn_citytile"]
            else:
                rewards[piece_id] += self.reward_map["turn_unit"]

                if lost_city[team]:
                    rewards[piece_id] += self.reward_map["death_city"]

                # resources (reward not negative)
                unit: Unit = pieces[piece_id]
                # check that the unit also existed last turn
                if unit.id in self.last_game_state["teamStates"][team]["units"].keys():
                    unit_last_turn = self.last_game_state["teamStates"][team]["units"][unit.id]
                    rewards[piece_id] += self.reward_map["wood_collected"] * max(
                        unit.cargo["wood"] - unit_last_turn.cargo["wood"],
                        0
                    )
                    rewards[piece_id] += self.reward_map["coal_collected"] * max(
                        unit.cargo["coal"] - unit_last_turn.cargo["coal"],
                        0
                    )
                    rewards[piece_id] += self.reward_map["uranium_collected"] * max(
                        unit.cargo["uranium"] - unit_last_turn.cargo["uranium"],
                        0
                    )

            # Collected resources
            rewards[piece_id] += self.reward_map["global_wood_collected"] * \
                                 (self.game_state.stats['teamStats'][team]['resourcesCollected']['wood'] -
                                  self.last_game_stats['teamStats'][team]['resourcesCollected']['wood'])
            rewards[piece_id] += self.reward_map["global_coal_collected"] * \
                                 (self.game_state.stats['teamStats'][team]['resourcesCollected']['coal'] -
                                  self.last_game_stats['teamStats'][team]['resourcesCollected']['coal'])
            rewards[piece_id] += self.reward_map["global_uranium_collected"] * \
                                 (self.game_state.stats['teamStats'][team]['resourcesCollected']['uranium'] -
                                  self.last_game_stats['teamStats'][team]['resourcesCollected']['uranium'])
            # Fuel
            rewards[piece_id] += self.reward_map["fuel_generated"] * \
                                 (self.game_state.stats['teamStats'][team]['fuelGenerated'] -
                                  self.last_game_stats['teamStats'][team]['fuelGenerated'])

            # Research Points
            if not self.last_game_state['teamStates'][team]['researched']['uranium']:
                rewards[piece_id] += self.reward_map["research_points"] * \
                                     (self.game_state.state['teamStates'][team]['researchPoints'] -
                                      self.last_game_state['teamStates'][team]['researchPoints'])

                # Reached coal/uranium research level
                for resource in ['coal', 'uranium']:
                    if self.game_state.state['teamStates'][team]['researched'][resource] and not \
                            self.last_game_state['teamStates'][team]['researched'][resource]:
                        rewards[piece_id] += self.reward_map[f"{resource}_researched"]

            if is_game_over:
                if team == 0:
                    rewards[piece_id] += self.reward_map["citytiles_end"] * city_tiles_team1
                    rewards[piece_id] -= self.reward_map["citytiles_end_opponent"] * city_tiles_team2
                else:
                    rewards[piece_id] += self.reward_map["citytiles_end"] * city_tiles_team2
                    rewards[piece_id] -= self.reward_map["citytiles_end_opponent"] * city_tiles_team1
                if team == winning_team:
                    rewards[piece_id] += self.reward_map["win"]

        team_average_reward = {}

        for team in [0, 1]:
            total_reward = 0
            N = 0
            for piece_id in rewards.keys():
                if piece_id[1] == team:
                    total_reward += rewards[piece_id]
                    N += 1
            if N > 0:
                team_average_reward[team] = total_reward / N
            else:
                team_average_reward[team] = 0

        if self.team_spirit > 0:
            for piece_id in rewards.keys():
                team = int(piece_id[1])
                rewards[piece_id] = round((1 - self.team_spirit) * rewards[piece_id] + \
                                          self.team_spirit * team_average_reward[team], 6)

        return rewards

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
            return f"p{team}_{piece.id}_{seed}"
        else:
            return f"p{team}_ct_{piece.get_tile_id()}_{seed}"

    @staticmethod
    def piece_id_to_unit_id(piece_id):
        # p0_u_1_2348
        return "_".join(piece_id.split("_")[1:3])

    def generate_obs(self):
        _map_player0 = generate_simple_map_obs(self.game_state, 0)
        _map_player1 = generate_simple_map_obs(self.game_state, 1)

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
                'map': Box(shape=(self.game_state.map.width, self.game_state.map.height, 9),
                           dtype=np.float32,
                           low=-float('inf'),
                           high=float('inf')
                           ),
                'game_state': Box(shape=(3,),
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
        map_obs = generate_simple_map_obs(game_state=env.game_state, team=0)
        game_obs = generate_simple_game_state_obs(game_state=env.game_state)

        obs, rewards, dones, infos = env.step(actions)
        done = env.game_state.match_over()

        if p_id in rewards:
            print(f"Reward: {rewards[p_id]}")
            print(f"Done: {dones[p_id]}")

    print("\nMatch over. End state:")
    env.render("cli")
