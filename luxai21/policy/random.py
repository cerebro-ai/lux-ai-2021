import numpy as np
from gym.spaces import Discrete, Dict, Box
from ray.rllib import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelWeights


class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs):
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        return [self.action_space.sample() for _ in obs_batch], \
               [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass


RandomWorkerPolicy = PolicySpec(
    policy_class=RandomPolicy,
    action_space=Discrete(9),
    observation_space=Dict(
        **{'map': Box(shape=(12, 12, 21),
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
                      low=0,
                      high=99999
                      ),
           'action_mask': Box(shape=(9,),
                              dtype=np.float64,
                              low=0,
                              high=1
                              )}),
    config={}
)
