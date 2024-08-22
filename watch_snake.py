from train import *
import time
import sys
import re

filename = sys.argv[1]

def watch_snake(model, num_episodes, board_size, max_steps):
    gen = re.search('-\d+', filename).group()[1:]
    done =  False
    for episode in range(num_episodes):
        game = Snake(board_size)
        state = game.get_state()
        returns = 0
        for step in range(max_steps):
            # Choose action
            action = choose_action(model, state)
            # Take action and observe next state and reward
            next_state, reward, done = game.step(action)
            print(f'generation: {gen}')
            print(game)

            returns += reward
            state = next_state

            if done:
                break
            time.sleep(0.02)

model = PolicyNetwork(12, 4)
model = torch.load(filename)
watch_snake(model, 50, 10, 100000)