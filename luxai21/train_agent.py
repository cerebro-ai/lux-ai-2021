from luxai21.agent.lux_agent import LuxAgent
from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv

n_games = 10

config = example_config.config

model = agent_model
agent1 = LuxPPOAgent(model)
agent2 = LuxPPOAgent(model)

agents = {
    "player_0": agent1,
    "player_1": agent2
}

env = LuxEnv(config)

for e in range(n_games):

    obs = env.reset()
    done = env.game_state.match_over()

    while not done:
        actions = {
            player: agent.generate_actions(obs["player"])
            for player, agent in agents.items()
        }
        obs, rewards, dones, infos = env.step(actions)

        for agent_id, agent in agents.items():
            agent.receive_reward(rewards[agent_id])

        done = env.game_state.match_over()



    for agent in agents.values():
        agent.match_over_callback()
