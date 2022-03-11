import util
from golf_env_continuous import GolfEnv
from src.random_agent import RandomAgent

if __name__ == '__main__':
    env = GolfEnv()
    agent = RandomAgent()

    # env.show_grayscale()
    # episode iteration
    for _ in range(1):
        (x, y, img, t) = env.reset()
        util.show_grayscale(img)

        for i in range(5):
            (s, a, r, (x, y, img, t)) = env.step(agent.step(), debug=True)
            util.show_grayscale(img)

        env.plot()
