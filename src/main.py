from golf_env import GolfEnv
from heuristic_agent import HeuristicAgent
from random_agent import RandomAgent

if __name__ == '__main__':
    env = GolfEnv()
    agent = RandomAgent()

    for _ in range(1):
        state = env.reset(regenerate_club_availability=True, max_timestep=-1)

        while True:
            state, r, term = env.step(agent.step(state), regenerate_heuristic_club_availability=True, debug=True)
            if term:
                break

        env.plot()
