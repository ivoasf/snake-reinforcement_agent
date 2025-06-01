#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from game.snake_game import SnakeGame


""" function to plot the game board in a png file """
def plot_board(file_name, board, text=None):
    plt.figure(figsize=(10,10))
    plt.imshow(board)
    plt.axis('off')

    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45,color = 'yellow')

    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


""" funtion to run a demo of the snake game """
def snake_demo(actions):
    game = SnakeGame(30, 30, border=1) # 30x30 board with a border of 1 pixel
    
    # reset the game and get the initial board state
    board, reward, done, info = game.reset() 
    plot_board('0.png', board, 'Start')

    action_name = {-1:'Turn Left', 0:'Straight Ahead', 1:'Turn Right'} # label each action

    # execute the actions and get the resulting board state
    for frame, action in enumerate(actions):
        board, reward, done, info = game.step(action)
        plot_board(f'{frame+1}.png', board, action_name[action])


"""
    -1 = turn left
    0  = straight ahead
    1  = turn right
"""
snake_demo([0, 1, 0, -1])
