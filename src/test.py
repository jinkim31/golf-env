from golf_env import GolfEnv
from heuristic_agent import HeuristicAgent
import util
from random_agent import RandomAgent

def main():
    env = GolfEnv()
    agent = RandomAgent()

    state = env.reset(max_timestep=100)

    env.step((0, 1), accurate_shots=True)
    env.step((90, 1), accurate_shots=True)
    env.step((45, 1), accurate_shots=True)
    env.plot()

if __name__ == '__main__':
    main()