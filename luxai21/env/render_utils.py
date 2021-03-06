from luxpythonenv.game.cell import Cell
from luxpythonenv.game.game_map import GameMap


def grid(row, col):
    """version with string concatenation"""
    sep = '\n' + '+---' * col + '+\n'
    return sep + ('|   ' * col + '|' + sep) * row


def cell_to_char(cell: Cell):
    # render city
    if cell.city_tile is not None:
        city_tile = cell.city_tile
        return "βΊοΈ" if city_tile.team == 0 else "π"

    # render worker
    if len(cell.units.keys()) > 0:
        unit = list(cell.units.values())[0]
        return "π" if unit.team == 0 else "π"

    # render resources
    if cell.resource is not None:
        resource = cell.resource
        if resource.type == "wood":
            return "π³"
        if resource.type == "coal":
            return "πͺ¨"
        if resource.type == "uranium":
            return "π"

    return "  "


def print_map(game_map: GameMap):
    row_sep = '\n' + '+----' * game_map.width + '+\n'

    string = row_sep

    for row in game_map.map:
        row_str = ''.join([f'| {cell_to_char(cell)} ' for cell in row] + ["|"])
        string += row_str
        string += row_sep

    print(string)
