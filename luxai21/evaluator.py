class Evaluator:

    def __init__(self, env, agent, num_games):
        self.env = env
        self.agent = agent
        self.num_games = num_games

    def get_win_rate(self, opponent):
        agents = {
            "player_0": self.agent1,
            "player_1": opponent
        }

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
