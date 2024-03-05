from simpleenvs.envs.discrete_rooms import DiscreteRoomEnvironment

from . import data

# Import room template files.
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


# XuFourRooms Variants
with pkg_resources.path(data, "xu_four_rooms_brtl.txt") as path:
    xu_four_rooms_brtl = path


class DiscreteXuFourRoomsBRTL(DiscreteRoomEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(xu_four_rooms_brtl, movement_penalty, goal_reward)


with pkg_resources.path(data, "xu_four_rooms_trbl.txt") as path:
    xu_four_rooms_trbl = path


class DiscreteXuFourRoomsTRBL(DiscreteRoomEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(xu_four_rooms_trbl, movement_penalty, goal_reward)


# NineRooms Variants
with pkg_resources.path(data, "nine_rooms_bltr.txt") as path:
    nine_rooms_bltr = path


class DiscreteNineRoomsBLTR(DiscreteRoomEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(nine_rooms_bltr, movement_penalty, goal_reward)


with pkg_resources.path(data, "nine_rooms_brtl.txt") as path:
    nine_rooms_brtl = path


class DiscreteNineRoomsBRTL(DiscreteRoomEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(nine_rooms_brtl, movement_penalty, goal_reward)


# Ramesh Maze Variants
with pkg_resources.path(data, "ramesh_maze_bltr.txt") as path:
    ramesh_maze_bltr = path


class DiscreteRameshMazeBLTR(DiscreteRoomEnvironment):
    def __init__(self, movement_penalty=-0.001, goal_reward=1):
        super().__init__(ramesh_maze_bltr, movement_penalty, goal_reward)
