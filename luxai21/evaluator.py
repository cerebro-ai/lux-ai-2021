from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env.lux_env import LuxEnv
import copy
from loguru import logger as log
import wandb
import os
from tqdm import tqdm


class Evaluator:

    def __init__(self, agent, config, num_games):
        self.env = LuxEnv(config)
        self.agent = agent
        self.num_games = num_games
        self.config = config
        self.loglevel = os.environ.get("LOGURU_LEVEL")

    def get_win_rate(self, opponent):
        agent = LuxPPOAgent(**self.config["agent"])
        agent.actor_critic = copy.deepcopy(self.agent.actor_critic)
        wins = []
        agents = {
            "player_0": agent,
            "player_1": opponent
        }
        if self.loglevel not in ["WARNING", "ERROR"]:
            turn_bar = tqdm(total=self.num_games, desc="Evaluation games played", ncols=90)

        for game in range(self.num_games):
            obs = self.env.reset()
            done = self.env.game_state.match_over()

            # GAME TURNS
            while not done:
                # 1. generate actions
                actions = {
                    player: agent.generate_actions(obs[player])
                    for player, agent in agents.items()
                }

                # 2. pass actions to env
                try:
                    obs, rewards, dones, infos = self.env.step(actions)
                except AttributeError as e:
                    pass

                # 4. check if game is over
                done = self.env.game_state.match_over()

            if self.loglevel not in ["WARNING", "ERROR"]:
                turn_bar.update(1)
            wins.append(1 if self.env.game_state.get_winning_team() == 0 else 0)

        win_rate = sum(wins) / len(wins)
        wandb.log({
            "win_rate": win_rate
        })
        log.debug(f"Win rate evaluation {win_rate}")

        return win_rate
