config = {
    "seed": 424,
    "wandb": {
        "entity": "cerebro-ai",
        "project": "luxai21",
        "notes": "First run with GNNs",
        "tags": ["GNNs", "Reward_func1"],
        "replay_every_x_games": 5
    },
    "game": {
        "width": 12,
        "height": 12,
    },
    "env": {
        "allow_carts": False,
    },
    "training": {
        "max_games": 10,
        "max_replay_buffer_size": 20000,
        "max_training_time": 30600,
        "save_checkpoint_every_x_games": 5
    },
    "agent": {
        "learning_rate": 0.001,
        "gamma": 0.95,
        "tau": 0.8,
        "batch_size": 80,  # two days
        "epsilon": 0.2,
        "epoch": 4,
        "entropy_weight": 0.005
    },
    "reward": {
        "BUILD_CITY_TILE": 0.01,  # reward for every new build city, will be negative if a city_tile vanishes
        "BUILD_WORKER": 0.01,
        "BUILD_CART": 0.01,

        "START_NEW_CITY": -0.005,  # we want to reward bigger cities because of cheaper upkeep

        "CITY_AT_END": 1,
        "UNIT_AT_END": 0.1,

        "GAIN_RESEARCH_POINT": 0.01,
        "RESEARCH_COAL": 0.1,
        "RESEARCH_URANIUM": 0.5,

        "WIN": 100,

        "ZERO_SUM": True
        # if true it will center the agent rewards around zero, and one agent will get a negative reward
    },
}
