import numpy as np
from luxpythonenv.game.cell import Cell
from luxpythonenv.game.city import City, CityTile
from luxpythonenv.game.constants import Constants
from luxpythonenv.game.game import Game
from luxpythonenv.game.game_constants import GAME_CONSTANTS
from luxpythonenv.game.position import Position


def find_all_resources(game_state):
    resource_tiles = []
    height = game_state.map.height
    width = game_state.map.width
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    return resource_tiles


def generate_map_state_matrix(game_state: Game):
    """
        Creates a (map height x map width x 17) matrix representing the current map state.
        :param game_state: current lux.game.Game object
        :return: np.ndarray containing the encoded map state
        """
    map_state = np.zeros((game_state.map.height, game_state.map.width, 17))
    fuel_normalizer = 1000

    resource_tiles = find_all_resources(game_state)
    for tile in resource_tiles:
        if tile.resource.type == Constants.RESOURCE_TYPES.WOOD:
            map_state[tile.pos.x][tile.pos.y][0] = tile.resource.amount / GAME_CONSTANTS["PARAMETERS"][
                "MAX_WOOD_AMOUNT"]
        elif tile.resource.type == Constants.RESOURCE_TYPES.COAL:
            map_state[tile.pos.x][tile.pos.y][1] = tile.resource.amount / GAME_CONSTANTS["PARAMETERS"][
                "MAX_WOOD_AMOUNT"]
        else:
            map_state[tile.pos.x][tile.pos.y][2] = tile.resource.amount / GAME_CONSTANTS["PARAMETERS"][
                "MAX_WOOD_AMOUNT"]

    for _, city in game_state.cities.items():
        city: City = city
        if city.team == 0:
            for tile in city.city_cells:
                city_tile: CityTile = tile.city_tile
                map_state[tile.pos.x][tile.pos.y][3] = 1
                map_state[tile.pos.x][tile.pos.y][5] = city_tile.cooldown / \
                                                       GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
                map_state[tile.pos.x][tile.pos.y][6] = city.fuel / len(city.city_cells)
                map_state[tile.pos.x][tile.pos.y][7] = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] - \
                                                       GAME_CONSTANTS["PARAMETERS"]["CITY_ADJACENCY_BONUS"] * \
                                                       city_tile.adjacent_city_tiles
                map_state[tile.pos.x][tile.pos.y][6] /= fuel_normalizer
                map_state[tile.pos.x][tile.pos.y][7] /= fuel_normalizer

        if city.team == 1:
            for tile in city.city_cells:
                map_state[tile.pos.x][tile.pos.y][4] = 1
                map_state[tile.pos.x][tile.pos.y][5] = tile.city_tile.cooldown / GAME_CONSTANTS["PARAMETERS"][
                    "CITY_ACTION_COOLDOWN"]
                map_state[tile.pos.x][tile.pos.y][6] = city.fuel / fuel_normalizer

    for unit in game_state.state["teamStates"][0]["units"].values():
        in_city: bool = True if game_state.map.get_cell_by_pos(unit.pos).city_tile is not None else False
        if unit.is_worker():
            map_state[unit.pos.x][unit.pos.x][8] += 1  # units can stack on a citytile
            map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['wood'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['coal'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][14] = unit.cargo['uranium'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][15] = unit.cooldown / \
                                                    GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][16] = 0 if in_city \
                                                    else GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][16] /= fuel_normalizer

        elif unit.is_cart():
            map_state[unit.pos.x][unit.pos.x][10] += 1
            map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['wood'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['coal'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][14] = unit.cargo['uranium'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][15] = unit.cooldown / \
                                                    GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"]
            map_state[unit.pos.x][unit.pos.y][16] = 0 if in_city \
                                                    else GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CART"]
            map_state[unit.pos.x][unit.pos.y][16] /= fuel_normalizer

    for unit in game_state.state["teamStates"][1]["units"].values():
        in_city: bool = True if game_state.map.get_cell_by_pos(unit.pos).city_tile is not None else False
        if unit.is_worker():
            map_state[unit.pos.x][unit.pos.x][9] += 1
            map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['wood'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['coal'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][14] = unit.cargo['uranium'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][15] = unit.cooldown / \
                                                    GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][16] = 0 if in_city \
                else GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER"]
            map_state[unit.pos.x][unit.pos.y][16] /= fuel_normalizer
        elif unit.is_cart():
            map_state[unit.pos.x][unit.pos.x][11] += 1
            map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['wood'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['coal'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][14] = unit.cargo['uranium'] / \
                                                    GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
            map_state[unit.pos.x][unit.pos.y][15] = unit.cooldown / \
                                                    GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"]
            map_state[unit.pos.x][unit.pos.y][16] = 0 if in_city \
                else GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CART"]
            map_state[unit.pos.x][unit.pos.y][16] /= fuel_normalizer

    for y in range(game_state.map.height):
        for x in range(game_state.map.width):
            cell = game_state.map.get_cell_by_pos(Position(x, y))
            map_state[cell.pos.x][cell.pos.y][17] = cell.road / GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
            map_state[cell.pos.x][cell.pos.y][18] = 1  # is map cell

    return map


def switch_map_matrix_player_view(map_matrix: np.ndarray):
    """
    Switch planes
        3, 4: city tiles
        7, 8: worker
        9,10: cart

    Args:
        map_matrix: numpy 18x32x32 array, player planes switched
    """
    map_copy = map_matrix.copy()
    map_copy[:, :, 3] = map_matrix[:, :, 4]
    map_copy[:, :, 4] = map_matrix[:, :, 3]

    map_copy[:, :, 7] = map_matrix[:, :, 8]
    map_copy[:, :, 8] = map_matrix[:, :, 7]

    map_copy[:, :, 9] = map_matrix[:, :, 10]
    map_copy[:, :, 10] = map_matrix[:, :, 9]

    return map_copy


def generate_game_state_matrix(game_state: Game, team: int):
    """
    Args:
        game_state: Game
        team: player_id, 0 or 1
    """

    """Get game state as numpy (1x5) array
        :param game_state:
        :return: Numpy array of shape (1x5)
        """
    city_normalizer = 10
    citytiles_normalizer = 100
    units_normalizer = 100
    total_fuel_normalizer = 10000
    research_points_normalizer = 100
    wood_normalizer = 10000
    coal_normalizer = 1000
    uranium_normalizer = 1000

    current_step = game_state.state['turn']
    game_progress = current_step / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
    day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
    days_until_night = np.maximum((day_length - current_step % 40) / day_length, 0)
    is_night = 1 if (current_step % 40) >= 30 else 0
    night_days_left = ((current_step % 40) - 30 if (current_step % 40) > 30 else 0) / 10
    team_cities = 0
    enemy_cities = 0
    for _, city in game_state.cities.items():
        if city.team == team:
            team_cities += 1
        else:
            enemy_cities += 1
    team_cities /= city_normalizer
    enemy_cities /= city_normalizer
    team_citytiles = game_state.stats['teamStats'][team]['cityTilesBuilt'] / citytiles_normalizer
    enemy_citytiles = game_state.stats['teamStats'][(team + 1) % 2]['cityTilesBuilt'] / citytiles_normalizer
    team_workers = game_state.stats['teamStats'][team]['workersBuilt'] / units_normalizer
    enemy_workers = game_state.stats['teamStats'][(team + 1) % 2]['workersBuilt'] / units_normalizer
    team_carts = game_state.stats['teamStats'][team]['cartsBuilt'] / units_normalizer
    enemy_carts = game_state.stats['teamStats'][(team + 1) % 2]['cartsBuilt'] / units_normalizer
    team_total_fuel = game_state.stats['teamStats'][team]['fuelGenerated'] / total_fuel_normalizer
    enemy_total_fuel = game_state.stats['teamStats'][(team + 1) % 2]['fuelGenerated'] / total_fuel_normalizer
    team_research_points = game_state.state["teamStates"][team]["researchPoints"] / research_points_normalizer
    enemy_research_points = game_state.state["teamStates"][(team + 1) % 2][
                                "researchPoints"] / research_points_normalizer
    team_wood = game_state.stats['teamStats'][team]['resourcesCollected']['wood'] / wood_normalizer
    team_coal = game_state.stats['teamStats'][team]['resourcesCollected']['coal'] / coal_normalizer
    team_uranium = game_state.stats['teamStats'][team]['resourcesCollected']['uranium'] / uranium_normalizer
    enemy_wood = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['wood'] / wood_normalizer
    enemy_coal = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['coal'] / coal_normalizer
    enemy_uranium = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['uranium'] / uranium_normalizer

    return np.array(
        [game_progress, days_until_night, is_night, night_days_left, team_cities, enemy_cities, team_citytiles,
         enemy_citytiles, team_workers, enemy_workers, team_carts, enemy_carts, team_total_fuel,
         enemy_total_fuel, team_research_points, enemy_research_points, team_wood, team_coal,
         team_uranium, enemy_wood, enemy_coal, enemy_uranium])


def generate_unit_states(game_state: Game, team: int):
    """
    Return a dictionary where the keys are the unit_id or citytile_id and the value the unit
    {
        'u1': {
            type:  0=worker, 1=cart, 2=city_tile
            state: [cooldown, cargo], [fuel, fuel_burn, cooldown]
            pos: (x, y)
            action_mask: Discrete(12)

    }

    """
