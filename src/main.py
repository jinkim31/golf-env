from src.golf_env_discrete import GolfEnvDiscrete
from src.heuristic_agent import HeuristicAgent
import util

if __name__ == '__main__':
    env = GolfEnvDiscrete()
    agent = HeuristicAgent()

    for _ in range(1):
        state = env.reset_randomized()
        #util.show_grayscale(state[0])

        # ((img, dist), r, term) = env.step((util.deg_to_rad(180), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(30), 100), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(21), 140), debug=True)
        # util.show_grayscale(img)
        # ((img, dist), r, term) = env.step((util.deg_to_rad(0), 100), debug=True)
        # util.show_grayscale(img)

        while True:
            state, r, term = env.step(agent.step(state), debug=True)
            #util.show_grayscale(state[0])
            if term:
                break

        env.plot()
