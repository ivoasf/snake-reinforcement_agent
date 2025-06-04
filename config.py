" board "
BORDER = 1
WIDTH = 14
HEIGHT = 14
SIZE = (WIDTH + (BORDER*2), HEIGHT + (BORDER*2))

" game "
NUM_ACTIONS = 3 # -1 (left), 0 (straight), 1 (right)
NUM_FRAMES = 3
INPUT_CHANNELS = NUM_FRAMES*3  # n frames, each with 3 channels (RGB)

" training "
LEARNING_RATE = 1e-4
BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor
TAU = 0.005 # update rate of the target network
