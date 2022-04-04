import numpy as np

from src.golf_env_discrete import GolfEnvDiscrete


class HeuristicAgent:

    def __init__(self):
        pass

    def step(self, state):
        distance = state[1]

        if distance > 300:
            club = np.random.randint(0, 5)
        elif distance > 200:
            club = np.random.randint(5, 7)
        elif distance > 100:
            club = np.random.randint(1, 19)
        elif distance > 70:
            club = np.random.randint(10, 19)
        else:
            club = np.random.randint(11, 19)

        return np.random.uniform(-45, 45), club
