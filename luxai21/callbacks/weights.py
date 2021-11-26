from loguru import logger

from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks


class UpdateWeightsCallback(DefaultCallbacks):
    win_rate_to_rotate = 0.7
    min_steps_between_updates = 10

    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        try:
            win_rate_0 = result["custom_metrics"]["win_rate_0"]
            win_rate_1 = result["custom_metrics"]["win_rate_1"]
            win_rate = (win_rate_1 + win_rate_0) / 2
            result["custom_metrics"]["win_rate"] = win_rate
        except KeyError:
            logger.warning("no win_rate. skip opponents callback")
            return

        if not hasattr(trainer, "steps_between_updates"):
            trainer.steps_between_updates = self.min_steps_between_updates
        else:
            if trainer.steps_between_updates > 0:
                trainer.steps_between_updates = trainer.steps_between_updates - 1

        if win_rate > self.win_rate_to_rotate:
            if trainer.steps_between_updates > 0:
                pass
            else:
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
                        new_weights[policy] = {
                            k: v.copy() for k, v in current_weights[source_policy].items()
                        }
                        logger.debug(f"Move weights: {source_policy} -> {policy}")
                    else:
                        new_weights[policy] = current_weights[policy]

                trainer.set_weights(new_weights)
                trainer.steps_between_updates = self.min_steps_between_updates

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
