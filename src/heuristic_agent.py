from . import golf_env
import numpy as np


class HeuristicAgent:

    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def step(self, state):
        dist_to_pin = state[1]
        club_n = len(golf_env.GolfEnv.CLUB_INFO)

        while True:
            club = np.random.randint(club_n)
            if golf_env.GolfEnv.CLUB_INFO[club][golf_env.GolfEnv.ClubInfoIndex.IS_DIST_PROPER](dist_to_pin):
                break

        return np.random.uniform(-45, 45), club
