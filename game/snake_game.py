#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import randint


class SnakeGame:
    def __init__(self, width, height, food_amount=1,
                 border=0, grass_growth=0, max_grass=0):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width, 3), dtype = np.float32)
        self.food_amount = food_amount
        self.border = border
        self.grass_growth = grass_growth
        self.grass = np.zeros((height, width)) + max_grass
        self.max_grass = max_grass
        self.reset()

    " create a new apple away from the snake "
    def create_apples(self):
        while len(self.apples) < self.food_amount:
            apple = ( randint(0, self.height-1), randint(0, self.width-1) )

            while apple in self.snake:
                apple = ( randint(0, self.height-1), randint(0, self.width-1) )

            self.apples.append(apple)

    " create a snake, size 3, at random position and orientation "
    def create_snake(self):
        x = randint( 5, self.width-5 )
        y = randint( 5, self.height-5 )
        self.direction = randint(0, 4)
        self.snake = []

        for i in range(5):
            if self.direction == 0:
                y = y+1
            elif self.direction == 1:
                x = x-1
            elif self.direction == 2:
                y = y-1
            elif self.direction == 3:
                x = x+1
            self.snake.append( (y,x) )

    " add one position to snake head (0=up, 1=right, 2=down, 3=left) "
    def grow_snake(self, d):
        y, x = self.snake[0]

        if d == 0:
            y = y-1
        elif d == 1:
            x = x+1
        elif d == 2:
            y = y+1
        else:
            x = x-1

        self.snake.insert(0, (y,x))

    " check if game is over by colliding with edge or itself "
    def check_collisions(self):
        x, y = self.snake[0]

        if (x == -1 or x == self.height 
            or y == -1 or y == self.width
            or (x,y) in self.snake[1:]):
            self.done = True

    """
        move snake/game one step 
        action can be -1 (turn left), 0 (continue), 1 (turn rignt)
    """
    def step(self, action):        
        direction = int(action)
        assert -1 <= direction <= 1        
        self.direction += direction

        if self.direction < 0:
            self.direction = 3
        elif self.direction > 3:
            self.direction = 0

        self.grow_snake(self.direction) 

        if self.snake[0] in self.apples:            
            self.apples.remove(self.snake[0])
            reward = 1
            self.create_apples()
        else:
            self.snake.pop()
            self.check_collisions()
            if self.done:
                reward = -1
            else:
                reward = 0
        if reward >= 0:
            x, y = self.snake[0]
            reward += self.grass[x,y]
            self.grass[x,y] = 0
            self.score += reward
            self.grass += self.grass_growth 
            self.grass[self.grass > self.max_grass] = self.max_grass
                
        return self.board_state(), reward, self.done, {'score':self.score}

    " easily get current state (score, apple, snake head and tail) "
    def get_state(self):        
        score = self.score
        apple = self.apples
        head = self.snake[0]
        tail = self.snake[1:]

        return score, apple, head, tail, self.direction
        
    " print the current board state "
    def print_state(self):
        for i in range(self.height):
            line = '.' * self.width
            for x, y in self.apples:
                if y == i:
                    line = line[:x] + 'A' + line[x+1:]
            for s in self.snake:
                x, y = s
                if y == i:
                    line = line[:x] + 'X' + line[x+1:]
            
            print(line)

    " to test: move the snake and print the game state "
    def test_step(self, direction):
        self.step(direction)
        self.print_state()

        if self.done:
            print("Game over! Score= ", self.score)
    
    " reset state "
    def reset(self):
        self.score = 0
        self.done = False
        self.create_snake()
        self.apples = []
        self.create_apples()
        self.grass[:,:] = self.max_grass
        
        return self.board_state(), 0, self.done, {'score':self.score}
    
    " render the environment "
    def board_state(self, mode='human', close=False):
        self.board[:,:,:] = 0

        if self.max_grass > 0:
            self.board[:,:,1] = self.grass/self.max_grass * 0.3
        if not self.done:
            x, y = self.snake[0]
            self.board[x,y,:] = 1
        for x, y in self.snake[1:]:
            self.board[x,y,0] = 1
        for x, y in self.apples: 
            self.board[x,y,1] = 1
        if self.border == 0:
            return self.board
        else:
            h, w, _ = self.board.shape
            board = np.full((h + self.border*2, w + self.border*2, 3), 0.5, np.float32)
            board[self.border:-self.border, self.border:-self.border] = self.board 
            return board


if __name__ == '__main__':
    game = SnakeGame(20, 20)
    game.print_state()
