from golf_env import GolfEnv
from heuristic_agent import HeuristicAgent
import util

if __name__ == '__main__':
    env = GolfEnv()
    agent = HeuristicAgent()

    for _ in range(1):
        state = env.reset(regenerate_club_availability=True)

        while True:
            state, r, term = env.step(agent.step(state), debug=True)
            if term:
                break

        env.plot()
