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
            episode.media["replay"] = wandb.Html(env.render("html"), inject=False)
            print("save replay")
        except AttributeError as e:
            print(f"Could not generate replay: {e}")
            print(f"Game match_over should be True:", {env.game_state.match_over()})

        # episode.hist_data["player_city_tiles"] = episode.user_data["player_city_tiles"]
        # episode.hist_data["opponent_city_tiles"] = episode.user_data["opponent_city_tiles"]
        # episode.hist_data["player_worker"] = episode.user_data["player_worker"]
        # episode.hist_data["opponent_worker"] = episode.user_data["player_city_tiles"]

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch,
                      **kwargs) -> None:
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):

        if "replay" in result["episode_media"].keys():
            # compute the score of each game, given end city_tiles and worker
            game_score = 1000 * np.array(result["hist_stats"]["end_player_city_tiles"]) + np.array(
                result["hist_stats"]["end_player_worker"])

            # Pick the best game
            best_game_index = np.argmax(game_score)
            print("log best game with score", game_score[best_game_index])
            result["episode_media"]["best_game"] = result["episode_media"]["replay"][best_game_index]

            # Pick the worst game
            worst_game_index = np.argmin(game_score)
            print("log worst game with score", game_score[best_game_index])
            result["episode_media"]["worst_game"] = result["episode_media"]["replay"][worst_game_index]

            # Delete all other replays
            del result["episode_media"]["replay"]

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
