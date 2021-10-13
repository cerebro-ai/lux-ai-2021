"""
Lux AI env following the DeepMind RL Environment API
https://github.com/deepmind/dm_env

"""
import dm_env
import numpy as np
from gym.spaces import Discrete, Dict, Box
from dm_env import specs
from pettingzoo.utils import agent_selector

UNIT_FOV = 3


class LuxEnv(dm_env.Environment):
    """
    Lux Multi Agent Environment following the specs of PettingZoo env
    """

    def observation_spec(self):
        return {
            'map_obs': specs.Array(shape=(18, 32, 32),
                                   dtype=np.float32,
                                   name="map_obs"
                                   ),
            'game_stats': specs.Array(shape=(22,),
                                      dtype=np.float32,
                                      name="game_stats"
                                      ),
            'unit_type': specs.DiscreteArray(3),
            'unit_obs': specs.Array(shape=(18, 2 * UNIT_FOV + 1, 2 * UNIT_FOV + 1),
                                    dtype=np.float32,
                                    name="unit_obs"
                                    ),
            'unit_stats': specs.Array(shape=(3,),
                                      dtype=np.float32,
                                      name="unit_stats"
                                      ),
            'action_mask': specs.Array(shape=(12,),
                                       dtype=np.int32,
                                       name="action_mask"
                                       ),
        }

    def action_spec(self):
        pass

    def __init__(self):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {'player_0': 0, 'player_1': 1}

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Discrete(12) for agent in self.possible_agents}
        self.observation_spaces = {agent: Dict({
            'map_obs': Box(shape=(18, 32, 32),
                           dtype=np.float32,
                           low=-float('inf'),
                           high=float('inf')
                           ),
            'game_stats': Box(shape=(22,),
                              dtype=np.float32,
                              low=float('-inf'),
                              high=float('inf')
                              ),
            'unit_type': Discrete(3),
            'unit_obs': Box(shape=(18, 2 * UNIT_FOV + 1, 2 * UNIT_FOV + 1),
                            dtype=np.float32,
                            low=float('-inf'),
                            high=float('inf')
                            ),
            'unit_stats': Box(shape=(3,),
                              dtype=np.float32,
                              low=float('-inf'),
                              high=float('inf')
                              ),
            'action_mask': Box(shape=(12,),
                               dtype=np.int,
                               low=0,
                               high=1
                               ),
        }) for agent in self.possible_agents}

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def reset(self):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def render(self, mode='human'):
        raise NotImplementedError()

    def state(self):
        pass
