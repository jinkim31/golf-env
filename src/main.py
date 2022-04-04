import numpy as np

from src import util
from src.golf_env_continuous import GolfEnvContinuous
from src.golf_env_discrete import GolfEnvDiscrete
from src.heuristic_agent import HeuristicAgent
from src.random_agent import RandomAgent

if __name__ == '__main__':
    env = GolfEnvDiscrete()
    agent = HeuristicAgent()

    for _ in range(10):
        state = env.reset()
        # util.show_grayscale(img)

        # ((img, dist), r, term) = env.step((util.deg_to_rad(180), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(30), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(21), 140), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(0), 100), debug=True)
        # util.show_grayscale(img)

        while True:
            state, r, term = env.step(agent.step(state), debug=True)
            if term:
                break

        env.plot()
