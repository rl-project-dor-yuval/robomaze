import numpy as np


class DemoBuffer:
    def __init__(self, size):
        self.size = size

        self.states = np.zeros((size, 5))
        self.actions = np.zeros((size, 3))
        self.rewards = np.zeros((size, 1))
        self.next_states = np.zeros((size, 5))
        self.dones = np.zeros((size, 1))
        self.values = np.zeros(size)

        self.is_full = False
        self.curr_idx = 0

    def add(self, state, action, reward, next_state, done, value):
        self.states[self.curr_idx] = state
        self.actions[self.curr_idx] = action
        self.rewards[self.curr_idx] = reward
        self.next_states[self.curr_idx] = next_state
        self.dones[self.curr_idx] = done
        self.values[self.curr_idx] = value

        if self.is_full:
            # set random curr index for the next sample
            self.curr_idx = np.random.randint(self.size)
        else:
            self.curr_idx += 1
            if self.curr_idx == self.size:
                self.is_full = True
                self.curr_idx = np.random.randint(self.size)

    def sample(self, batch_size):
        if self.is_full:
            idx = np.random.randint(self.size, size=batch_size)
        else:
            idx = np.random.randint(self.curr_idx, size=batch_size)

        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx], \
               self.values[idx]

