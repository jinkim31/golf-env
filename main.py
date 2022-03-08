import math

import numpy
import numpy as np

import util
from golf_env import GolfEnv

if __name__ == '__main__':
    env = GolfEnv()

    for _ in range(1):
        env.step((util.deg_to_rad(0), 30))
        env.step((util.deg_to_rad(45), 30))
        env.step((util.deg_to_rad(90), 30))
        env.reset()

    env.plot()
