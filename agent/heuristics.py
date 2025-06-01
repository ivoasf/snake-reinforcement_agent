"""
    An implementation of a heuristic for the snake game.
    MinDistanceHeuristic is a simple heuristic that tries to minimize the distance to the apple.
"""

from game.game_wrapper import SnakeGameWrapper


TURN_LEFT = 0
GO_STRAIGHT = 1
TURN_RIGHT = 2

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Heuristic:
    def get_action(self, game: SnakeGameWrapper) -> int:
        raise NotImplementedError


class MinDistanceHeuristic(Heuristic):
    def get_action(self, game: SnakeGameWrapper) -> int:
        _, apples, head, _, direction = game.game.get_state() 

        target = apples[0]

        # head and target are tuples of (y, x) coordinates
        dy = target[0] - head[0]
        dx = target[1] - head[1]

        if direction == UP:
            if dx < 0:
                return TURN_LEFT  # apple is to the left
            elif dx > 0:
                return TURN_RIGHT  # apple is to the right
            elif dy > 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == RIGHT:
            if dy < 0:
                return TURN_RIGHT  # apple is downwards
            elif dy > 0:
                return TURN_LEFT  # apple is upwards
            elif dx > 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == DOWN:
            if dx < 0:
                return TURN_RIGHT  # apple is to the right
            elif dx > 0:
                return TURN_LEFT  # apple is to the left
            elif dy < 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == LEFT:
            if dy < 0:
                return TURN_LEFT  # apple is downwards
            elif dy > 0:
                return TURN_RIGHT  # apple is upwards
            elif dx < 0:
                return GO_STRAIGHT  # apple is straight ahead

        # if none of the above conditions are met, return GO_STRAIGHT as a fallback
        return GO_STRAIGHT
