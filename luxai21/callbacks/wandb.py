import os
from multiprocessing import Queue
from typing import Optional, Dict, List

from ray.tune.integration.wandb import _WandbLoggingProcess, _set_api_key, _clean_log, _WANDB_QUEUE_END
from ray.tune.logger import LoggerCallback
from ray.tune.trial import Trial

"""
This is a copy of the WandbLoggerCallback from ray tune.

The only difference is:
- introduce wandb entity as parameter

"""


class WandbLoggerCallback(LoggerCallback):
    """WandbLoggerCallback

    Weights and biases (https://www.wandb.ai/) is a tool for experiment
    tracking, model optimization, and dataset versioning. This Ray Tune
    ``LoggerCallback`` sends metrics to Wandb for automatic tracking and
    visualization.

    Args:
        project (str): Name of the Wandb project. Mandatory.
        group (str): Name of the Wandb group. Defaults to the trainable
            name.
        api_key_file (str): Path to file containing the Wandb API KEY. This
            file only needs to be present on the node running the Tune script
            if using the WandbLogger.
        api_key (str): Wandb API Key. Alternative to setting ``api_key_file``.
        excludes (list): List of metrics that should be excluded from
            the log.
        log_config (bool): Boolean indicating if the ``config`` parameter of
            the ``results`` dict should be logged. This makes sense if
            parameters will change during training, e.g. with
            PopulationBasedTraining. Defaults to False.
        **kwargs: The keyword arguments will be pased to ``wandb.init()``.

    Wandb's ``group``, ``run_id`` and ``run_name`` are automatically selected
    by Tune, but can be overwritten by filling out the respective configuration
    values.

    Please see here for all other valid configuration settings:
    https://docs.wandb.ai/library/init

    Example:

    .. code-block:: python

        from ray.tune.logger import DEFAULT_LOGGERS
        from ray.tune.integration.wandb import WandbLoggerCallback
        tune.run(
            train_fn,
            config={
                # define search space here
                "parameter_1": tune.choice([1, 2, 3]),
                "parameter_2": tune.choice([4, 5, 6]),
            },
            callbacks=[WandbLoggerCallback(
                project="Optimization_Project",
                api_key_file="/path/to/file",
                log_config=True)])

    """

    # Do not log these result keys
    _exclude_results = ["done", "should_checkpoint"]

    # Use these result keys to update `wandb.config`
    _config_results = [
        "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
        "pid", "date"
    ]

    _logger_process_cls = _WandbLoggingProcess

    def __init__(self,
                 entity: str,
                 project: str,
                 group: Optional[str] = None,
                 api_key_file: Optional[str] = None,
                 api_key: Optional[str] = None,
                 excludes: Optional[List[str]] = None,
                 log_config: bool = False,
                 **kwargs):
        self.entity = entity
        self.project = project
        self.group = group
        self.api_key_path = api_key_file
        self.api_key = api_key
        self.excludes = excludes or []
        self.log_config = log_config
        self.kwargs = kwargs

        self._trial_processes: Dict["Trial", _WandbLoggingProcess] = {}
        self._trial_queues: Dict["Trial", Queue] = {}

    def setup(self):
        self.api_key_file = os.path.expanduser(self.api_key_path) if \
            self.api_key_path else None
        _set_api_key(self.api_key_file, self.api_key)

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.trainable_name if trial else None

        # remove unpickleable items!
        config = _clean_log(config)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=trial_name,
            resume=True,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            entity=self.entity,
            project=wandb_project,
            config=config)
        wandb_init_kwargs.update(self.kwargs)

        self._trial_queues[trial] = Queue()
        self._trial_processes[trial] = self._logger_process_cls(
            queue=self._trial_queues[trial],
            exclude=exclude_results,
            to_config=self._config_results,
            **wandb_init_kwargs)
        self._trial_processes[trial].start()

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_processes:
            self.log_trial_start(trial)

        result = _clean_log(result)
        self._trial_queues[trial].put(result)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        self._trial_queues[trial].put(_WANDB_QUEUE_END)
        self._trial_processes[trial].join(timeout=10)

        del self._trial_queues[trial]
        del self._trial_processes[trial]

    def __del__(self):
        for trial in self._trial_processes:
            if trial in self._trial_queues:
                self._trial_queues[trial].put(_WANDB_QUEUE_END)
                del self._trial_queues[trial]
            self._trial_processes[trial].join(timeout=2)
            del self._trial_processes[trial]
