from loguru import logger
import tempfile
from copy import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker, MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID
from ray.tune import Callback
from wandb import util

from luxai21.env.lux_ma_env import LuxMAEnv


class UpdateWeightsCallback(DefaultCallbacks):
    win_rate_to_rotate = 0.6

    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        try:
            win_rate = result["custom_metrics"]["win_rate"]
        except KeyError:
            logger.warning("no win_rate. skip opponents callback")
            return

        if win_rate > self.win_rate_to_rotate:
            current_weights = trainer.get_weights()
            new_weights = {}
            for policy in current_weights.keys():
                if "opponent" in policy:
                    rank = int(policy.split("_")[-1])
                    if rank == 1:
                        # get the newest weights from the learned worker
                        source_policy = "player_worker"
                    else:
                        # rank N gets weights from N-1 and so on
                        source_policy = f"opponent_worker_{rank - 1}"
                    new_weights[policy] = current_weights[source_policy]
                    logger.debug(f"Move weights: {source_policy} -> {policy}")
                else:
                    new_weights[policy] = current_weights[policy]

            trainer.set_weights(new_weights)

            # log current opponent_level

            if hasattr(trainer, "opponent_level"):
                trainer.opponent_level += 1
            else:
                trainer.opponent_level = 1

            logger.info("Opponents updated")

        # log current opponent_level
        if hasattr(trainer, "opponent_level"):
            opponent_level = trainer.opponent_level
        else:
            opponent_level = 0

        logger.debug(f"Opponent level: {opponent_level}")
        result["custom_metrics"]["opponent_level"] = opponent_level
