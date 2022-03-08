import numpy
import numpy as np

import util
from golf_env import GolfEnv

if __name__ == '__main__':
    env = GolfEnv()

    for _ in range(1000):
        env.step((util.deg_to_rad(0), 100))
        env.step((util.deg_to_rad(45), 100))
        env.step((util.deg_to_rad(90), 100))
        env.reset()

    env.plot()

    tf = util.transform_2d(100, 200, util.deg_to_rad(47))
    inv = util.inv_transform_2d(tf)

    print(tf)
    print(inv)
    print(np.dot(tf, inv))

