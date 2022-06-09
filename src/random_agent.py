import numpy as np
from . import golf_env


class RandomAgent:
    # noinspection PyMethodMayBeStatic
    def step(self, state):
        return np.random.uniform(-30, 30), np.random.randint(len(golf_env.GolfEnv.SKILL_MODEL))
