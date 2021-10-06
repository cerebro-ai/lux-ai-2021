import sys
import time
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game import Game
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.
    Args:
        team ([type]): [description]
        unit_id ([type]): [description]
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


def append_position_layer(game_state_matrix: np.ndarray, entity) -> np.ndarray:
    layer = np.zeros((32, 32))
    layer[entity.pos.x][entity.pos.y] = 1
    return np.dstack((game_state_matrix, layer))


def create_map_state_matrix(game_state: Game) -> np.ndarray:
    """
    Creates a (map height x map width x 17) matrix representing the current map state.
    The map is encoded in the following way:
        0. Wood (float)
        1. Coal (float)
        2. Uranium (float)
        3. own_city_tile
        4. enemy_city_tile
        5. city_cooldown
        6. fuel city
        7. own_worker
        8. enemy_worker
        9. own_cart
       10. enemy_cart
       11. cargo.wood
       12. cargo.coal
       13. cargo.uranium
       14. unit_cooldown (shared across worker & cart)
       15. road_lvl (float)
       16. is_valid_map (this is needed because we pad to 32x32)
    :param game_state: current lux.game.Game object
    :return: np.ndarray containing the encoded map state
    """
    map_state = np.zeros((game_state.map.height, game_state.map.width, 17))
    pad_width = (32 - game_state.map.width) // 2

    resource_tiles = find_all_resources(game_state)
    for tile in resource_tiles:
        if tile.resource.type == Constants.RESOURCE_TYPES.WOOD:
            map_state[tile.pos.x][tile.pos.y][0] = tile.resource.amount
        elif tile.resource.type == Constants.RESOURCE_TYPES.COAL:
            map_state[tile.pos.x][tile.pos.y][1] = tile.resource.amount
        else:
            map_state[tile.pos.x][tile.pos.y][2] = tile.resource.amount

    for _, city in game_state.cities.items():
        if city.team == 0:
            for tile in city.city_cells:
                map_state[tile.pos.x][tile.pos.y][3] = 1
                map_state[tile.pos.x][tile.pos.y][5] = tile.city_tile.cooldown
                map_state[tile.pos.x][tile.pos.y][6] = city.fuel
        if city.team == 1:
            for tile in city.city_cells:
                map_state[tile.pos.x][tile.pos.y][4] = 1
                map_state[tile.pos.x][tile.pos.y][5] = tile.city_tile.cooldown
                map_state[tile.pos.x][tile.pos.y][6] = city.fuel

    for unit in game_state.state["teamStates"][0]["units"].values():
        if unit.type == 0:
            map_state[unit.pos.x][unit.pos.x][7] += 1  # units can stack on a citytile
        elif unit.type == 1:
            map_state[unit.pos.x][unit.pos.x][9] += 1
        map_state[unit.pos.x][unit.pos.y][11] = unit.cargo['wood']
        map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['coal']
        map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['uranium']
        map_state[unit.pos.x][unit.pos.y][14] = unit.cooldown

    for unit in game_state.state["teamStates"][1]["units"].values():
        if unit.is_worker():
            map_state[unit.pos.x][unit.pos.x][8] += 1
        elif unit.is_cart():
            map_state[unit.pos.x][unit.pos.x][10] += 1
        map_state[unit.pos.x][unit.pos.y][11] = unit.cargo['wood']
        map_state[unit.pos.x][unit.pos.y][12] = unit.cargo['coal']
        map_state[unit.pos.x][unit.pos.y][13] = unit.cargo['uranium']
        map_state[unit.pos.x][unit.pos.y][14] = unit.cooldown

    for y in range(game_state.map.height):
        for x in range(game_state.map.width):
            cell = game_state.map.get_cell_by_pos(Position(x, y))
            map_state[cell.pos.x][cell.pos.y][15] = cell.road
            map_state[cell.pos.x][cell.pos.y][16] = 1  # is map cell

    map_padded = np.pad(map_state, [(pad_width,), (pad_width,), (0,)], mode="constant", constant_values=0)
    return map_padded


def get_game_state_matrix(game_state: Game, team):
    """Get game state as numpy (1x5) array
    :param game_state:
    :return: Numpy array of shape (1x5)
    """
    current_step = game_state.turn
    wood = 0
    coal = 0
    uranium = 0
    night = 1 if (game_state.turn % 40) > 30 else 0
    for tile in find_all_resources(game_state):
        if tile.resource.type == Constants.RESOURCE_TYPES.WOOD:
            wood += tile.resource.amount
        elif tile.resource.type == Constants.RESOURCE_TYPES.COAL:
            coal += tile.resource.amount
        elif tile.resource.type == Constants.RESOURCE_TYPES.URANIUM:
            uranium += tile.resource.amount
        else:
            # this should actually never happen
            print("not recognised tile type", tile.resource.type)
            pass
    research_points = game_state.state["teamStates"][team]["researchPoints"]
    return np.array([current_step, night, wood, coal, uranium])


########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            # Transfer to nearby cart
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),
            # Transfer to nearby worker
            SpawnCityAction,
            PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        self.observation_shape = (3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        obs = create_map_state_matrix(game)

        entity = None
        if unit is not None:
            entity = unit
        else:
            entity = city_tile

        obs = append_position_layer(obs, entity)
        obs = obs.flatten()
        # obs = np.hstack(obs, get_game_state_matrix(game, team)) # TODO Fix bugs here

        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile != None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.
        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1

        rewards = {}

        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ((fuel_collected - self.fuel_collected_last) / 20000)
        self.fuel_collected_last = fuel_collected

        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

            '''
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            '''

        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.
        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
