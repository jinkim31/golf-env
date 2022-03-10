from abc import *


class Env(metaclass=ABCMeta):
    def __init__(self, action_space_size, state_space_size):
        """
        :param action_space_size:
        :param state_space_size:
        """
        self.__action_space_size = action_space_size
        self.__state_space_size = state_space_size
        print('Env init(' + str(self.__action_space_size) + "," + str(self.__state_space_size) + ")")

    @property
    def action_space_size(self):
        return self.__action_space_size

    @property
    def state_space_size(self):
        return self.__state_space_size

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        :param action: tuple of action
        :return: tuple of transition (s,a,r,s',termination)
        """
        pass

    @abstractmethod
    def plot(self):
        pass
