import numpy as np
import random

PENALTY = -10
REWARD = 100
# DEFAULT = -0.05
DEFAULT = 0

class Snake:
    def __init__(self, size):
        self.size = size
        self.positions = {
            (size // 2 - 1, size // 2): None,
            (size // 2, size // 2): None,
            (size // 2 + 1, size // 2): None
        }
        self.head = (size // 2 + 1, size // 2)
        self.apple = self.make_random_apple(None)
        self.done = False
        self.score = 0

    def make_random_apple(self, newhead):
        locs = []
        for i in range(self.size):
            for j in range(self.size):
                loc = (i, j)
                if not loc in self.positions and loc != newhead:
                    locs.append(loc)
        if len(locs) > 0:
            return random.choice(locs)
        else:
            return (0,0)
    
    # L = 0, R = 1, U = 2, D = 3
    def step(self, action):
        # get new head location
        newhead_r = self.head[0]
        newhead_c = self.head[1]
        if action == 0:
            newhead_c += -1
        elif action == 1:
            newhead_c += 1
        elif action == 2:
            newhead_r += -1
        elif action == 3:
            newhead_r += 1
        newhead = (newhead_r, newhead_c)
        reward = DEFAULT
        # check for snake hitting walls or itself
        if self.is_collision(newhead):
            self.done = True
            return (self.get_state(), PENALTY, self.done)
        # check for eating apple
        elif newhead == self.apple:
            self.score += 1
            self.apple = self.make_random_apple(newhead)
            reward = REWARD
        else:
            self.positions.pop(next(iter(self.positions)))
        self.head = newhead
        self.positions[newhead] = None
        return (self.get_state(), reward, self.done)
    
    def is_collision(self, loc):
        if (loc[0] < 0 or loc[0] >= self.size or loc[1] < 0 or loc[1] >= self.size) or loc in self.positions:
            return True
        return False
    
    # basic state
    # def get_state(self):
    #     l = (self.head[0], self.head[1] - 1)
    #     r = (self.head[0], self.head[1] + 1)
    #     u = (self.head[0] - 1, self.head[1])
    #     d = (self.head[0] + 1, self.head[1])
    #     state = [self.is_collision(l),
    #              self.is_collision(r),
    #              self.is_collision(u),
    #              self.is_collision(d),
    #              self.apple[0] < self.head[0],
    #              self.apple[0] > self.head[0],
    #              self.apple[1] < self.head[1],
    #              self.apple[1] > self.head[1]]
    #     return np.array(state, dtype=bool)
    
    # location state
    def get_state(self):
        l = (self.head[0], self.head[1] - 1)
        r = (self.head[0], self.head[1] + 1)
        u = (self.head[0] - 1, self.head[1])
        d = (self.head[0] + 1, self.head[1])
        state = [self.is_collision(l),
                 self.is_collision(r),
                 self.is_collision(u),
                 self.is_collision(d),
                 self.apple[0] < self.head[0],
                 self.apple[0] > self.head[0],
                 self.apple[1] < self.head[1],
                 self.apple[1] > self.head[1],
                 1.0 / (self.head[0] + 1),
                 1.0 / (self.size - self.head[0]),
                 1.0 / (self.head[1] + 1),
                 1.0 / (self.size - self.head[1])]
        return np.array(state, dtype=float)
    
    # location and size of snake
    # def get_state(self):
    #     l = (self.head[0], self.head[1] - 1)
    #     r = (self.head[0], self.head[1] + 1)
    #     u = (self.head[0] - 1, self.head[1])
    #     d = (self.head[0] + 1, self.head[1])
    #     state = [self.is_collision(l),
    #              self.is_collision(r),
    #              self.is_collision(u),
    #              self.is_collision(d),
    #              self.apple[0] < self.head[0],
    #              self.apple[0] > self.head[0],
    #              self.apple[1] < self.head[1],
    #              self.apple[1] > self.head[1],
    #              1.0 / (self.head[0] + 1),
    #              1.0 / (self.size - self.head[0]),
    #              1.0 / (self.head[1] + 1),
    #              1.0 / (self.size - self.head[1]),
    #              1.0 / (self.score + 1)]
    #     return np.array(state, dtype=float)

    @staticmethod
    def neg_to_inf(x):
        if x < 0:
            x = np.inf
        return x

    def __str__(self):
        hrow = ['X'] * (self.size + 2)
        hrow.append(' score: %d\n' % self.score)
        hrow_str = ''.join(hrow)
        board = [hrow_str]
        for i in range(self.size):
            row = ['X']
            for j in range(self.size):
                if (i, j) == self.apple:
                    row.append('A')
                elif (i, j) in self.positions:
                    row.append('O')
                else:
                    row.append(' ')
            row.append('X\n')
            board.append(''.join(row))
        board.append(hrow_str)
        return ''.join(board)
        
