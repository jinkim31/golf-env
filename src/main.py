import numpy as np

from src import util
from src.golf_env_continuous import GolfEnvContinuous
from src.golf_env_discrete import GolfEnvDiscrete
from src.random_agent import RandomAgent

if __name__ == '__main__':
    env = GolfEnvDiscrete()
    agent = RandomAgent()

    for _ in range(1):
        (img, dist) = env.reset()
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
            ((img, dist), r, term) = env.step((np.random.uniform(-90, 90), np.random.randint(0, 24)), debug=True)
            if term:
                break

        env.plot()
