from . import golf_env
from . import util
from . import random_agent

def main():
    env = golf_env.GolfEnv()
    agent = random_agent.RandomAgent()

    state = env.reset(max_timestep=100)

    env.step((0, 1), accurate_shots=True)
    env.step((90, 1), accurate_shots=True)
    env.step((45, 1), accurate_shots=True)
    env.plot()

if __name__ == '__main__':
    main()