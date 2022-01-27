import tempfile
from copy import copy
from pathlib import Path
from typing import Dict, Optional

import loguru
import numpy as np
import wandb
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID
from wandb import util

from luxai21.env.lux_ma_env import LuxMAEnv


class MetricsCallback(DefaultCallbacks):
    log_replays = False

    def get_player_team(self, episode):
        return episode.episode_id % 2

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
        player_team = self.get_player_team(episode)
        opponent_team = (player_team + 1) % 2
        player_city_tiles = len(env.game_state.get_teams_citytiles(player_team))
        opponent_city_tiles = len(env.game_state.get_teams_citytiles(opponent_team))
        player_worker = len([x for x in env.game_state.get_teams_units(player_team).values() if x.is_worker()])
        opponent_worker = len([x for x in env.game_state.get_teams_units(opponent_team).values() if x.is_worker()])

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

        player_team = self.get_player_team(episode)

        end_player_city_tiles = episode.user_data["player_city_tiles"][-1]
        end_opponent_city_tiles = episode.user_data["opponent_city_tiles"][-1]
        end_player_worker = episode.user_data["player_worker"][-1]
        end_opponent_worker = episode.user_data["opponent_worker"][-1]

        episode.hist_data["end_player_city_tiles"] = [end_player_city_tiles]
        episode.hist_data["end_opponent_city_tiles"] = [end_opponent_city_tiles]
        episode.hist_data["end_player_worker"] = [end_player_worker]
        episode.hist_data["end_opponent_worker"] = [end_opponent_worker]

        episode.custom_metrics[f"game_won_{player_team}"] = 1 if env.game_state.get_winning_team() == player_team else 0

        if self.log_replays:
            try:
                # store the replay in a tmp file, which is currently not cleared!!!
                _, filepath = tempfile.mkstemp("_lux-replay_" + util.generate_id() + ".html")
                with Path(filepath).open("w") as f:
                    f.write(env.render("html"))
                episode.media["replay"] = wandb.Html(filepath, inject=False)
            except AttributeError as e:
                print(f"Could not generate replay: {e}")
                print(f"Game match_over should be True:", {env.game_state.match_over()})

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch,
                      **kwargs) -> None:
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):

        """
        If this part throws an KeyError than your rollout_fragment_length is possibly to small
        and no whole episode was collected and on_epi
        """

        try:

            n_replays = result["episodes_this_iter"]

            # compute the score of each game, given end city_tiles and worker
            train_iteration_city_tiles = np.array(result["hist_stats"]["end_player_city_tiles"][-n_replays:])
            train_iteration_worker = np.array(result["hist_stats"]["end_player_worker"][-n_replays:])
            game_score = 1000 * train_iteration_city_tiles + train_iteration_worker

            # Pick the best game
            best_game_index = np.argmax(game_score)
            print("log best game with score", game_score[best_game_index])
            if self.log_replays:
                print("best game replay", result["episode_media"]["replay"][-n_replays:][best_game_index]._path)
                result["episode_media"]["best_game"] = copy(
                    result["episode_media"]["replay"][-n_replays:][best_game_index])

            # store the number of city_tiles of the best game
            result["custom_metrics"]["best_game_city_tiles"] = train_iteration_city_tiles[best_game_index]

            # Pick the worst game
            worst_game_index = np.argmin(game_score)
            print("log worst game with score", game_score[worst_game_index])
            if self.log_replays:
                result["episode_media"]["worst_game"] = copy(
                    result["episode_media"]["replay"][-n_replays:][worst_game_index])

            # store the number of city_tiles of the worst game
            result["custom_metrics"]["worst_game_city_tiles"] = train_iteration_city_tiles[worst_game_index]

            # mean_city_tiles
            result["custom_metrics"]["mean_player_city_tiles"] = train_iteration_city_tiles.mean()
            result["custom_metrics"]["mean_player_worker"] = train_iteration_worker.mean()

            # Delete all other replays
            if self.log_replays:
                result["episode_media"]["replay"] = []

            # Rename metric game_won_mean to win_rate and remove max and min metrics
            if "game_won_0_mean" in result["custom_metrics"]:
                result["custom_metrics"]["win_rate_0"] = result["custom_metrics"]["game_won_0_mean"]

                del result["custom_metrics"]["game_won_0_mean"]
                del result["custom_metrics"]["game_won_0_min"]
                del result["custom_metrics"]["game_won_0_max"]

            if "game_won_1_mean" in result["custom_metrics"]:
                result["custom_metrics"]["win_rate_1"] = result["custom_metrics"]["game_won_1_mean"]

                del result["custom_metrics"]["game_won_1_mean"]
                del result["custom_metrics"]["game_won_1_min"]
                del result["custom_metrics"]["game_won_1_max"]

        except KeyError as e:
            loguru.logger.warning("Looks like the rollout contains no complete episode. skip logging")
            loguru.logger.warning(e)


def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                      result: dict, **kwargs) -> None:
    pass


def on_postprocess_trajectory(
        self, *, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str,
        policy_id: str, policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch], **kwargs):
    pass
