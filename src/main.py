import util
from golf_env_continuous import GolfEnv
from src.random_agent import RandomAgent

if __name__ == '__main__':
    env = GolfEnv()

    for _ in range(1):
        (img, dist) = env.reset()
        util.show_grayscale(img)

        ((img, dist), r, term) = env.step((util.deg_to_rad(180), 100), debug=True)
        util.show_grayscale(img)
        ((img, dist), r, term) = env.step((util.deg_to_rad(30), 100), debug=True)
        util.show_grayscale(img)
        ((img, dist), r, term) = env.step((util.deg_to_rad(21), 140), debug=True)
        util.show_grayscale(img)
        ((img, dist), r, term) = env.step((util.deg_to_rad(0), 100), debug=True)
        util.show_grayscale(img)

        env.plot()
