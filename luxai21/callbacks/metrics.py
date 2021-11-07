import tempfile
from copy import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID
from ray.tune import Callback
from wandb import util

from luxai21.env.lux_ma_env import LuxMAEnv


class MetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:

        episode.user_data["player_city_tiles"] = []
        episode.user_data["opponent_city_tiles"] = []
        episode.user_data["player_worker"] = []
        episode.user_data["opponent_worker"] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        # take the first environment (multiple envs)
        env: LuxMAEnv = base_env.envs[0]
        player_city_tiles = len(env.game_state.get_teams_citytiles(0))
        opponent_city_tiles = len(env.game_state.get_teams_citytiles(1))
        player_worker = len([x for x in env.game_state.get_teams_units(0).values() if x.is_worker()])
        opponent_worker = len([x for x in env.game_state.get_teams_units(1).values() if x.is_worker()])

        episode.user_data["player_city_tiles"].append(player_city_tiles)
        episode.user_data["opponent_city_tiles"].append(opponent_city_tiles)
        episode.user_data["player_worker"].append(player_worker)
        episode.user_data["opponent_worker"].append(opponent_worker)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        env: LuxMAEnv = base_env.envs[0]
        end_player_city_tiles = episode.user_data["player_city_tiles"][-1]
        end_opponent_city_tiles = episode.user_data["opponent_city_tiles"][-1]
        end_player_worker = episode.user_data["player_worker"][-1]
        end_opponent_worker = episode.user_data["opponent_worker"][-1]

        episode.hist_data["end_player_city_tiles"] = [end_player_city_tiles]
        episode.hist_data["end_opponent_city_tiles"] = [end_opponent_city_tiles]
        episode.hist_data["end_player_worker"] = [end_player_worker]
        episode.hist_data["end_opponent_worker"] = [end_opponent_worker]

        episode.custom_metrics["game_won"] = 1 if env.game_state.get_winning_team() == 0 else 0

        try:
            _, filepath = tempfile.mkstemp("_lux-replay_" + util.generate_id() + ".html")
            with Path(filepath).open("w") as f:
                f.write(env.render("html"))
            episode.media["replay"] = wandb.Html(filepath, inject=False)
        except AttributeError as e:
            print(f"Could not generate replay: {e}")
            print(f"Game match_over should be True:", {env.game_state.match_over()})

        # episode.hist_data["player_city_tiles"] = episode.user_data["player_city_tiles"]
        # episode.hist_data["opponent_city_tiles"] = episode.user_data["opponent_city_tiles"]
        # episode.hist_data["player_worker"] = episode.user_data["player_worker"]
        # episode.hist_data["opponent_worker"] = episode.user_data["player_city_tiles"]

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch,
                      **kwargs) -> None:
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):

        if "replay" in result["episode_media"].keys():
            n_replays = result["episodes_this_iter"]
            print(f"Number collected replays: {n_replays}")

            # compute the score of each game, given end city_tiles and worker
            game_score = 1000 * np.array(result["hist_stats"]["end_player_city_tiles"][-n_replays:]) + np.array(
                result["hist_stats"]["end_player_worker"][-n_replays:])

            # Pick the best game
            best_game_index = np.argmax(game_score)
            print("log best game with score", game_score[best_game_index])
            print("best game replay", result["episode_media"]["replay"][-n_replays:][best_game_index]._path)
            result["episode_media"]["best_game"] = copy(result["episode_media"]["replay"][-n_replays:][best_game_index])

            # store the number of city_tiles of the best game
            result["custom_metrics"]["best_game_city_tiles"] = \
                result["hist_stats"]["end_player_city_tiles"][-n_replays:][best_game_index]

            # Pick the worst game
            worst_game_index = np.argmin(game_score)
            print("log worst game with score", game_score[worst_game_index])
            result["episode_media"]["worst_game"] = copy(
                result["episode_media"]["replay"][-n_replays:][worst_game_index])

            # store the number of city_tiles of the worst game
            result["custom_metrics"]["worst_game_city_tiles"] = \
                result["hist_stats"]["end_player_city_tiles"][-n_replays:][worst_game_index]

            # Delete all other replays
            result["episode_media"]["replay"] = []

        # Rename metric game_won_mean to win_rate and remove other metrics
        if "game_won_mean" in result["custom_metrics"]:
            result["custom_metrics"]["win_rate"] = result["custom_metrics"]["game_won_mean"]

            del result["custom_metrics"]["game_won_mean"]
            del result["custom_metrics"]["game_won_min"]
            del result["custom_metrics"]["game_won_max"]

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
