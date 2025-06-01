#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from snake_game import SnakeGame


class SnakeGameWrapper:
    def __init__(self, game: SnakeGame, num_frames: int):
        self.game = game
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def reset(self):
        state, reward, done, info = self.game.reset()
        for _ in range(self.num_frames):
            self.frames.append(state)
        return self.get_state(), reward, done, info

    def step(self, action):
        state, reward, done, info = self.game.step(action)
        self.frames.append(state)
        return self.get_state(), reward, done, info

    def get_state(self):
        return self.__stack_frames()

    def get_last_state(self):
        return self.frames[-1]

    def __stack_frames(self):
        return np.stack(list(self.frames), axis=3)
    

if __name__ == "__main__":
    game = SnakeGame(20, 20)
    wrapper = SnakeGameWrapper(game, num_frames=4)

    state, reward, done, info = wrapper.reset()
    print("Shape of the queued state (frames):", state.shape)

    # output: (20, 20, 3, 4)
    # 20x20 is the grid size, 3 is the color channels, and 4 is the number of frames stacked

    for i in range(5):
        state, reward, done, info = wrapper.step(0) 
        print(f"Step {i+1}, Shape: {state.shape}")
