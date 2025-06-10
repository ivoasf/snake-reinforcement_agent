"""
    An implementation of a heuristic for the snake game.
    ImprovedHeuristic is designed to navigate the snake towards the nearest apple avoiding collisions.
"""

import config
from game.game_wrapper import SnakeGameWrapper


class Heuristic:
    def get_action(self, game: SnakeGameWrapper) -> int:
        raise NotImplementedError


class ImprovedHeuristic(Heuristic):
    def get_action(self, game: SnakeGameWrapper) -> int:
        _, apples, head, body, direction = game.game.get_state()
        target = apples[0]

        def next_pos(dir_change):
            new_dir = (direction + dir_change) % 4

            dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][new_dir]

            return (head[0] + dy, head[1] + dx)

        def is_safe(pos):
            y, x = pos
            width, height = config.SIZE

            if not (0 <= y < height and 0 <= x < width):
                return False
            
            return pos not in body
        
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        options = {
            0: next_pos(-1), # left
            1: next_pos(0),  # straight ahead
            2: next_pos(1),  # right
        }

        safe_actions = [a for a, pos in options.items() if is_safe(pos)]

        if not safe_actions:
            return 1 # straight ahead

        best = min(safe_actions, key=lambda a: manhattan(options[a], target))
        return best 
