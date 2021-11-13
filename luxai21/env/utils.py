from typing import Optional, Dict, Union, Tuple
import wandb

import numpy as np
from luxpythonenv.game.actions import TransferAction
from luxpythonenv.game.cell import Cell
from luxpythonenv.game.city import City, CityTile
from luxpythonenv.game.constants import Constants
from luxpythonenv.game.game import Game
from luxpythonenv.game.game_constants import GAME_CONSTANTS
from luxpythonenv.game.position import Position
from luxpythonenv.game.unit import Unit


def get_piece_id(team: int, piece: Union[CityTile, Unit]):
    if hasattr(piece, "cargo"):
        # is unit
        return f"p{team}_{piece.id}_{piece.rand_id}"
    else:
        return f"p{team}_ct_{piece.get_tile_id()}_{piece.rand_id}"


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


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.
    Args:
        game
        team ([type]): [description]
        unit_id ([type]): [description]
        unit
        target_type_restriction
        **kwarg
    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (u.get_cargo_space_left() >= resource_amount and
                                        target_unit.get_cargo_space_left() >= resource_amount):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif (target_unit.get_cargo_space_left() >= resource_amount):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif (u.get_cargo_space_left() > target_unit.get_cargo_space_left()):
                                    # Change targets, because neither target can accept all our resources and
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


def generate_map_state_matrix(game_state: Game):
    """
        Creates a (map height x map width x 17) matrix representing the current map state.
        :param game_state: current lux.game.Game object
        :return: np.ndarray containing the encoded map state
        """
    map_state = np.zeros((game_state.map.height, game_state.map.width, 21))
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

            # DIRECTIONAL EMBEDDING
            map_state[cell.pos.x][cell.pos.y][19] = cell.pos.x / (game_state.map.width - 1)
            map_state[cell.pos.x][cell.pos.y][20] = cell.pos.y / (game_state.map.height - 1)

    return map_state


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
    team_researched_coal = game_state.state["teamStates"][team]["researched"]["coal"]
    team_researched_uranium = game_state.state["teamStates"][team]["researched"]["uranium"]
    enemy_research_points = game_state.state["teamStates"][(team + 1) % 2][
                                "researchPoints"] / research_points_normalizer
    enemy_researched_coal = game_state.state["teamStates"][(team + 1) % 2]["researched"]["coal"]
    enemy_researched_uranium = game_state.state["teamStates"][(team + 1) % 2]["researched"]["uranium"]
    team_wood = game_state.stats['teamStats'][team]['resourcesCollected']['wood'] / wood_normalizer
    team_coal = game_state.stats['teamStats'][team]['resourcesCollected']['coal'] / coal_normalizer
    team_uranium = game_state.stats['teamStats'][team]['resourcesCollected']['uranium'] / uranium_normalizer
    enemy_wood = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['wood'] / wood_normalizer
    enemy_coal = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['coal'] / coal_normalizer
    enemy_uranium = game_state.stats['teamStats'][(team + 1) % 2]['resourcesCollected']['uranium'] / uranium_normalizer

    return np.array(
        [game_progress, days_until_night, is_night, night_days_left, team_cities, enemy_cities, team_citytiles,
         enemy_citytiles, team_workers, enemy_workers, team_carts, enemy_carts, team_total_fuel,
         enemy_total_fuel, team_research_points, team_researched_coal, team_researched_uranium, enemy_research_points,
         enemy_researched_coal, enemy_researched_uranium, team_wood, team_coal, team_uranium, enemy_wood, enemy_coal,
         enemy_uranium])


def generate_unit_states(game_state: Game, map_state: np.ndarray, team: int, config):
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
    states = {}
    game_state_array = generate_game_state_matrix(game_state, team)

    for _, city in game_state.cities.items():
        if city.team == team:
            for cell in city.city_cells:
                city_tile = cell.city_tile
                states[get_piece_id(team, city_tile)] = {
                    "type": 2,
                    "pos": np.array([cell.pos.x, cell.pos.y]),
                    "action_mask": get_action_mask(game_state, team, None, city_tile, config),
                    "map": map_state,
                    "mini_map": generate_mini_map(map_state, (cell.pos.x, cell.pos.y), config["fov"]),
                    "game_state": game_state_array
                }

    for unit in game_state.state["teamStates"][team]["units"].values():
        states[get_piece_id(team, unit)] = {
            "type": 0 if unit.is_worker() else 1,
            "pos": np.array([unit.pos.x, unit.pos.y]),
            "action_mask": get_action_mask(game_state, team, unit, None, config),
            "map": map_state,
            "mini_map": generate_mini_map(map_state, (unit.pos.x, unit.pos.y), config["fov"]),
            "game_state": game_state_array
        }
    return states


def generate_mini_map(map: np.ndarray, pos: Tuple, fov: int):
    """
    Generate a small cutout of the map around the given position with fov steps in each direction

    Args:
        map: the whole game map with [x,y,features]
        pos: Position where to center around the mini map
        fov: Field of view, how many tiles in each direction to include

    """
    # first we pad the map 0
    map_padded = np.pad(map, [(fov,), (fov,), (0,)], mode="constant", constant_values=0)
    x, y = pos[0] + fov, pos[1] + fov
    # cutout
    mini_map = map_padded[x - fov: x + fov + 1, y - fov:y + fov + 1, :]

    return mini_map


def get_action_mask(game_state: Game, team: int, unit: Optional[Unit], city_tile: Optional[CityTile], config):
    """
    actions:
    --- unit
    0. do nothing
    1. move north
    2. move west
    3. move south
    4. move east
    5. transfer to worker (smart transfer)
    6. transfer to cart (smart transfer)
    7. spawn city
    8. pillage
    --- city
    9.  do nothing
    10. spawn worker
    11. spawn cart
    12. research
    """

    if unit is not None:
        action_mask = np.zeros(9)

        action_mask[0] = 1  # always allow to do nothing

        # check if can act
        if not unit.can_act():
            return action_mask

        def is_enemy_city(x, y):
            cell = game_state.map.get_cell(x, y)
            if cell.is_city_tile():
                if cell.city_tile.team != team:
                    return True
            return False

        # MOVEMENT
        # Check boarder & enemy city_tile
        # NORTH
        if unit.pos.y != 0:
            if not is_enemy_city(unit.pos.x, unit.pos.y - 1):
                action_mask[1] = 1
        # WEST
        if unit.pos.x != 0:
            if not is_enemy_city(unit.pos.x - 1, unit.pos.y):
                action_mask[2] = 1
        # SOUTH
        if unit.pos.y != (game_state.map.height - 1):
            if not is_enemy_city(unit.pos.x, unit.pos.y + 1):
                action_mask[3] = 1
        # EAST
        if unit.pos.x != (game_state.map.width - 1):
            if not is_enemy_city(unit.pos.x + 1, unit.pos.y):
                action_mask[4] = 1

        # TRANSFER
        cell = game_state.map.get_cell_by_pos(unit.pos)
        adj_cells = game_state.map.get_adjacent_cells(cell)
        for cell in adj_cells:
            if cell.has_units():
                units = cell.units
                for _, unit in units.items():
                    # Check if unit in team
                    if unit.team == team:
                        # a. Check if worker nearby
                        if unit.type == 0:
                            action_mask[5] = 1
                            break
                        # b. Check if cart nearby
                        else:
                            if config["allow_carts"]:
                                action_mask[6] = 1
                            break

        # BUILD CITY & PILLAGE
        if unit.is_worker():
            if unit.can_build(game_state.map):

                # check that unit is not on a city tile
                cell = game_state.map.get_cell_by_pos(unit.pos)
                if cell.city_tile is None:
                    action_mask[7] = 1

                    if game_state.map.get_cell_by_pos(unit.pos).road > 0:
                        action_mask[8] = 1

    elif city_tile is not None:
        action_mask = np.zeros(4)

        # do nothing
        action_mask[0] = 1

        if not city_tile.can_act():
            return action_mask

        if city_tile.can_build_unit():
            if get_unit_count(game_state.state, team) < get_city_tile_count(game_state.cities, team):
                action_mask[1] = 1
                if config["allow_carts"]:
                    action_mask[2] = 1

        if city_tile.can_research():
            action_mask[3] = 1
    else:
        raise Exception("unit and city_tile both None")

    return action_mask


def get_city_count(game_state_cities: Dict, team: int):
    count = 0
    for city in game_state_cities.values():
        if city.team == team:
            count += 1
    return count


def get_city_tile_count(game_state_cities: Dict, team: int):
    count = 0
    for city in game_state_cities.values():
        if city.team == team:
            for cell in city.city_cells:
                city_tile = cell.city_tile
                if city_tile is not None:
                    count += 1
    return count


def get_unit_count(game_state: Dict, team: int):
    return len(game_state["teamStates"][team]["units"])


def get_worker_count(game_state: Dict, team: int):
    count = 0
    for unit in game_state["teamStates"][team]["units"].values():
        if unit.is_worker():
            count += 1
    return count


def get_cart_count(game_state: Dict, team: int):
    count = 0
    for unit in game_state["teamStates"][team]["units"].values():
        if unit.is_cart():
            count += 1
    return count


def log_and_get_citytiles_game_end(game_state: Game):
    # TODO Split according to map size
    citytiles_end = get_city_tile_count(game_state.cities, 0)
    wandb.log({
        'Citytiles_end_player_one': citytiles_end,
    })
    return citytiles_end
