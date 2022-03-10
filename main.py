import util
import numpy as np
from golf_env_continuous import GolfEnv

if __name__ == '__main__':
    env = GolfEnv()

    # episode iteration
    for _ in range(10):
        env.reset()
        env.step((util.deg_to_rad(0), 30))
        env.step((util.deg_to_rad(45), 30))
        env.step((util.deg_to_rad(-45), 30))
        env.plot()
