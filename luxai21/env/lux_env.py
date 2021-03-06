"""
Lux AI env following the PettingZoo ParallelEnv
"""

import copy
import json
import random
from functools import partial
from pathlib import Path
from typing import List, Mapping, Tuple, Any

import wandb
from gym.spaces import Discrete, Dict, Box
from luxpythonenv.game.actions import MoveAction, SpawnCityAction, PillageAction, SpawnWorkerAction, SpawnCartAction, \
    ResearchAction
from luxpythonenv.game.constants import LuxMatchConfigs_Default
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector

from luxai21.env.utils import *


class LuxEnv(ParallelEnv):
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
        self.reward_config = config["reward"]
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

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents
        self.agent_name_mapping = {'player_0': 0, 'player_1': 1}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.turn = 0

        self.action_map = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),  # 1
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),  # 2
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            SpawnCityAction,
            PillageAction,
            None,  # City do nothing
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction  # 12
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
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

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
                'player0': {
                    'u1': 7,
                    'c2': 2
                    },
                'player1': {
                    'u3': 4,
                    'c2': 1,
                    }
                }
        Returns:
            observations, rewards, dones, infos
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        game_actions = self.translate_actions(actions)
        is_game_done = self.game_state.run_turn_with_actions(actions=game_actions)
        rewards = self.compute_rewards(self.reward_config)

        self.last_game_state = copy.deepcopy(self.game_state.state)
        self.last_game_cities = copy.deepcopy(self.game_state.cities)

        observations = self.generate_obs()

        infos = {agent: {} for agent in self.agents}
        dones = {agent: is_game_done for agent in self.agents}

        self.turn += 1

        return observations, rewards, dones, infos

    def translate_actions(self, actions: Mapping[str, Mapping[str, int]]) -> List:
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

        for agent, pieces in actions.items():
            team = self.agent_name_mapping[agent]
            for piece_id, action_id in pieces.items():
                # city_tiles have no id, thus we find them by position
                if piece_id.startswith("ct"):
                    ident, x_str, y_str = piece_id.split("_")
                    cell = self.game_state.map.get_cell(int(x_str), int(y_str))
                    city_tile = cell.city_tile
                    if city_tile is None:
                        raise Exception(f"city_tile could not be found for {piece_id}")

                    action_class = self.action_map[action_id]
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
                    action = self.action_map[action_id](
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

    def compute_rewards(self, reward_config: dict) -> dict:
        """
        Args:
            reward_config: Dict with the specific reward values
        Returns:
            A dict with the respective rewards of every agent/player for the current game_state in comparison to
            the last_game_state

        reward_config = {

            "BUILD_CITY_TILE": 0.01,  # reward for every new build city, will be negative if a city_tile vanishes
            "BUILD_WORKER": 0.01,
            "BUILD_CART": 0.01,

            "START_NEW_CITY": -0.005,  # we want to reward bigger cities because of cheaper upkeep

            "CITY_AT_END": 1,
            "UNIT_AT_END": 0.1,

            "GAIN_RESEARCH_POINT": 0.01,
            "RESEARCH_COAL": 0.1,
            "RESEARCH_URANIUM": 0.5,

            "WIN": 100,

            "ZERO_SUM": True
            # if true it will center the agent rewards around zero, and one agent will get a negative reward
        }
        """

        rewards = np.zeros(2)

        for i, agent in enumerate(self.agents):

            rewards[i] += reward_config["TURN"]

            # reward new cities
            delta_city_tiles = get_city_tile_count(self.game_state.cities, i) - get_city_tile_count(
                self.last_game_cities, i)
            rewards[i] += delta_city_tiles * reward_config["BUILD_CITY_TILE"]

            # reward new worker
            delta_worker = get_worker_count(self.game_state.state, i) - get_worker_count(self.last_game_state, i)
            rewards[i] += delta_worker * reward_config["BUILD_WORKER"]

            # reward new cart
            delta_cart = get_cart_count(self.game_state.state, i) - get_cart_count(self.last_game_state, i)
            rewards[i] += delta_cart * reward_config["BUILD_CART"]

            # reward new city
            delta_city = get_city_count(self.game_state.cities, i) - get_city_count(self.last_game_cities, i)
            rewards[i] += delta_city * reward_config["START_NEW_CITY"]

            # research
            delta_research_points = self.game_state.state["teamStates"][i]["researchPoints"] - \
                                    self.last_game_state["teamStates"][i]["researchPoints"]
            rewards[i] = delta_research_points * reward_config["GAIN_RESEARCH_POINT"]

            if not self.last_game_state["teamStates"][i]["researched"]["coal"]:
                if self.game_state.state["teamStates"][i]["researched"]["coal"]:
                    rewards += reward_config["RESEARCH_COAL"]

            if not self.last_game_state["teamStates"][i]["researched"]["uranium"]:
                if self.game_state.state["teamStates"][i]["researched"]["uranium"]:
                    rewards += reward_config["RESEARCH_URANIUM"]

            # check if game over
            if self.game_state.match_over():
                rewards[i] += get_city_tile_count(self.game_state.cities, i) * reward_config["CITY_AT_END"]
                rewards[i] += len(self.game_state.get_teams_units(i)) * reward_config["UNIT_AT_END"]

                if i == self.game_state.get_winning_team():
                    rewards[i] += reward_config["WIN"]

        if reward_config["ZERO_SUM"]:
            rewards = rewards - np.mean(rewards)

        return {
            self.agents[i]: rewards[i] for i in range(len(self.agents))
        }

    def generate_obs(self):
        _map_player0 = generate_map_state_matrix(self.game_state)
        _map_player1 = switch_map_matrix_player_view(_map_player0)

        game_state_player0 = generate_game_state_matrix(self.game_state, team=0)
        game_state_player1 = generate_game_state_matrix(self.game_state, team=1)

        unit_states_player0 = generate_unit_states(self.game_state, team=0, config=self.env_config)
        unit_states_player1 = generate_unit_states(self.game_state, team=1, config=self.env_config)

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
            '_map': Box(shape=(18, self.game_state.map.width, self.game_state.map.height),
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
                })
                for unit in self.game_state.get_teams_units(i)
            }
        }) for i in [0, 1]}

    @property
    def action_spaces(self):
        return {agent: Discrete(12) for agent in self.agents}


if __name__ == '__main__':
    import example_config

    env = LuxEnv(config=example_config.config)

    use_wandb = True

    if use_wandb:
        wandb.init(project="luxai21", entity="cerebro-ai")

    obs = env.reset()

    while not env.game_state.match_over():
        actions = {
            "player_0": {
                "u_1": 1
            },
            "player_1": {
                "u_2": 3
            }
        }
        obs, rewards, dones, infos = env.step(actions)

    # with open("test.html", "w") as f:
    # html = env.render("html")
    # f.write(html)
    if use_wandb:
        wandb.log({"LuxEnv": wandb.Html(env.render("html"), inject=False)})
    print(env.turn)

    config = {
        'algorithm': "{'name': 'PPO', 'config': {'num_workers': 4, 'num_envs_per_worker': 1, 'gamma': 0.99, 'rollout_fragment_length': 256, 'train_batch_size': 1024, 'num_sgd_iter': 10, 'lr': 0.0002, 'sgd_minibatch_size': 256, 'batch_mode': 'complete_episodes', 'entropy_coeff': 3e-05}}",
        'env': "{'env': 'lux_ma_env', 'env_config': {'game': {'height': 12, 'width': 12}, 'env': {'allow_carts': False, 'fov': 2}, 'team_spirit': 0, 'reward': {'move': 0, 'transfer': 0, 'build_city': 1, 'pillage': 0, 'build_worker': 1, 'build_cart': 0.1, 'research': 1, 'wood_collected': 0.0, 'coal_collected': 0.0, 'uranium_collected': 0.0, 'turn_unit': 0.1, 'turn_citytile': 0.1, 'death_city': -1, 'fuel_generated': 0, 'research_points': 0, 'coal_researched': 0, 'uranium_researched': 0, 'global_wood_collected': 0.0, 'global_coal_collected': 0.0, 'global_uranium_collected': 0.0, 'win': 10, 'citytiles_end': 0, 'citytiles_end_opponent': 0}, 'save_replay': {'local_every_x': 0, 'wandb_every_x': 0}}}",
        'model': "{'worker': {'custom_model': 'worker_model_v2', 'custom_model_config': {'use_meta_node': True, 'map_model': {'input_dim': 21, 'hidden_dim': 128, 'output_dim': 256}, 'mini_map_model': {'input_dim': 21, 'hidden_dim': 128, 'output_dim': 256}, 'game_state_model': {'input_dim': 26, 'output_dim': 64}, 'embedding_size': 512, 'lstm': {'hidden_size': 512}, 'policy': {'hidden_size_1': 128, 'hidden_size_2': 32, 'hidden_size_3': 16, 'output_size': 9}, 'value': {'hidden_size_1': 128, 'hidden_size_2': 32}}}}",
        'debug': False, 'verbose': 1, 'stop': "{'timesteps_total': 5000}", 'checkpoint_at_end': True,
        'checkpoint_freq': 30, 'local_dir': '~/ray_results', 'metrics': "{'log_replays': False}",
        'weights': "{'win_rate_to_update': 0.7}",
        'wandb': "{'entity': 'cerebro-ai', 'project': 'luxai21', 'group': 'debug', 'notes': 'testing new model v2 architecture', 'tags': ['GNN', 'Dev', 'Rllib'], 'mode': 'online'}"}
