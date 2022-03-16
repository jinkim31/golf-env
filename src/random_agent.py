import numpy as np


class RandomAgent:
    def step(self):
        return np.random.uniform(-np.pi/4, np.pi/4), np.random.uniform(10, 100)
