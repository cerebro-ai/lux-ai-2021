from collections import deque
from typing import Dict, Tuple

import numpy as np
from hydra import compose, initialize
from ray.rllib.utils.typing import MultiAgentDict

from luxai21.env.lux_ma_env import LuxMAEnv


class Stack:
    """
    Push puts items into a stack
    Get gets a list of the items of the length stack_size.
    Queue is initialized with zero like arrays in the beginning
    """

    def __init__(self, stack_size: int):
        self.stack_size = stack_size
        self.stack: Dict[str, deque] = {}

    def push(self, key, item):
        if key in self.stack.keys():
            self.stack[key].appendleft(item)

        else:
            # new queue
            queue = deque(maxlen=self.stack_size)
            # fill queue with zero items
            zero_item = np.zeros_like(item)
            for _ in range(self.stack_size):
                queue.appendleft(zero_item)
            queue.appendleft(item)
            self.stack[key] = queue

    def get(self, key):
        if key in self.stack.keys():
            return list(self.stack[key])
        else:
            raise AttributeError(f"unknown key {key}")

    def reset(self):
        self.stack = {}


class StackedLuxMAEnv(LuxMAEnv):
    """
    Observations are buffered and stacked to the size stack_size
    The returned observation are stacked along the last dimension and padded with 0 if items are not available
    """

    def __init__(self, config: Dict = None):
        self.stack_size = config["stack_size"]
        assert self.stack_size >= 1, "Stack size should be larger then one"

        self.stack = Stack(self.stack_size)

        super(StackedLuxMAEnv, self).__init__(config)

    def step(self, action_dict: MultiAgentDict) \
            -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        turn_obs, rewards, dones, infos = super(StackedLuxMAEnv, self).step(action_dict)

        obs = self._get_stacked_obs(turn_obs)

        return obs, rewards, dones, infos

    def _get_stacked_obs(self, turn_obs) -> MultiAgentDict:
        # bootstrap dict with stacked obs
        obs = {key: {} for key in turn_obs.keys()}

        # iterate over current obs
        for piece_id, piece_obs in turn_obs.items():
            for obs_key, obs_item in piece_obs.items():
                key = f"{piece_id}_{obs_key}"
                # append obs to stack
                self.stack.push(key, obs_item)

                # get obs as list
                obs_list = self.stack.get(key)

                # stack along last dimension
                axis = obs_list[0].ndim - 1
                stacked_obs = np.concatenate(obs_list, axis=axis)
                obs[piece_id][obs_key] = stacked_obs

        return obs

    def reset(self) -> dict:
        # empty stacks and fill with zero value
        self.stack.reset()
        turn_obs = super(StackedLuxMAEnv, self).reset()
        stacked_obs = self._get_stacked_obs(turn_obs)
        return stacked_obs


if __name__ == '__main__':
    # test the stack
    stack_size = 3
    stack = Stack(stack_size)

    item = np.random.random((3,))
    key = "item_key"

    stack.push(key, item)

    obs = stack.get(key)

    print(np.concatenate(obs, axis=obs[0].ndim - 1).shape)

    #
    initialize(config_path="../conf")
    config = compose(config_name="config")
    stacked_env = StackedLuxMAEnv(config.env.env_config)
    obs = stacked_env.reset()

    print(obs)
    actions = {
        piece_id: 0
        for piece_id in obs.keys()
    }
    obs, _, _, _ = stacked_env.step(actions)

    print(obs)

    obs, _, _, _ = stacked_env.step(actions)
    print(obs)
    pass
