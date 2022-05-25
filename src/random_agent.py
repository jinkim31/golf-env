import numpy as np
from . import golf_env


class RandomAgent:
    def step(self, state):
        return np.random.uniform(-30, 30), np.random.randint(len(GolfEnv.CLUB_INFO))
