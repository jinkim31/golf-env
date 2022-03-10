import numpy as np


class RandomAgent:
    def step(self):
        return np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(10, 30)
