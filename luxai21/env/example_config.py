config = {
    "game": {
        "width": 12,
        "height": 12,
        "seed": 21
    },
    "agent": {
        "allow_carts": False
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
    }

}