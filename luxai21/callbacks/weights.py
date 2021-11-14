from loguru import logger

from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks


class UpdateWeightsCallback(DefaultCallbacks):
    win_rate_to_rotate = 0.7

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
