import matplotlib.pyplot as plt
import numpy as np
import util
import env


class GolfEnv(env.Env):
    START_X = 100
    START_Y = 100
    VAR_X = 5
    VAR_Y = 10

    def __init__(self):
        super().__init__(2, 2)  # action size:2(angle,club), state size:2(x,y)
        self.__state = None
        self.__initial_state = (self.START_X, self.START_Y)
        self.__marker_x = []
        self.__marker_y = []
        self.reset()

    def step(self, action):
        """
        steps simulator
        :param action: tuple of action(angle, club)
        :return: tuple of transition (s,a,r,s',term)
        s:before state(x,y) a:action, r:reward, s':new state(x',y'), term:termination
        """

        rng = np.random.default_rng()

        shoot = np.array([[action[1], 0]]) + rng.normal(size=2, scale=[self.VAR_X, self.VAR_Y])
        delta = np.dot(util.rotation_2d(action[0]), shoot.transpose()).transpose()
        self.__state = self.__state + delta.squeeze()

        self.__marker_x.append(self.__state[0])
        self.__marker_y.append(self.__state[1])

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.__marker_x, self.__marker_y, s=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.show()

    def reset(self):
        self.__state = self.__initial_state
        self.__marker_x.append(self.__state[0])
        self.__marker_y.append(self.__state[1])
