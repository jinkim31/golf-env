import math

import matplotlib.pyplot as plt
import numpy as np
import util
import env
import cv2


class GolfEnv(env.Env):
    IMG_URL = "resources/img.png"
    IMG_SIZE_X = 100
    IMG_SIZE_Y = 100
    START_X = 10
    START_Y = 10
    VAR_X = 1
    VAR_Y = 3
    PIN_X = 50
    PIN_Y = 90
    STATE_IMAGE_WIDTH = 20
    STATE_IMAGE_HEIGHT = 40
    STATE_IMAGE_OFFSET_HEIGHT = -4

    def __init__(self):
        super().__init__(2, 2)  # action size:2(angle,club), state size:2(x,y)
        self.__state = None
        self.__initial_state = (self.START_X, self.START_Y)
        self.__marker_x = []
        self.__marker_y = []
        self.__marker_end_x = []
        self.__marker_end_y = []
        self.__test_x = []
        self.__test_y = []
        self.img = cv2.cvtColor(cv2.imread(self.IMG_URL), cv2.COLOR_BGR2RGB)
        self.reset()
        self.generate_state_img()

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

        self.generate_state_img()

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, self.IMG_SIZE_X])
        plt.ylim([0, self.IMG_SIZE_Y])
        plt.imshow(plt.imread(self.IMG_URL), extent=[0, self.IMG_SIZE_X, 0, self.IMG_SIZE_Y])
        plt.scatter(self.PIN_X, self.PIN_Y, s=500, marker='x', color='black')
        plt.scatter(self.START_X, self.START_Y, s=200, color='black')
        plt.quiver(self.__marker_x, self.__marker_y, self.__marker_end_x, self.__marker_end_y)
        plt.scatter(self.__test_x, self.__test_y, s=0.1, color='black')
        plt.show()

    def reset(self):
        self.__state = self.__initial_state

    def generate_state_img(self):
        # save data to plot
        angle_to_pin = math.atan2(self.PIN_Y - self.__state[1], self.PIN_X - self.__state[0])
        arrow = np.dot(util.rotation_2d(angle_to_pin), np.array([[1, 0]]).transpose())
        self.__marker_x.append(self.__state[0])
        self.__marker_y.append(self.__state[1])
        self.__marker_end_x.append(arrow[0, 0])
        self.__marker_end_y.append(arrow[1, 0])

        # get tf between fixed frame and moving frame (to use p0 = t01*p1)
        t01 = util.transform_2d(self.__state[0], self.__state[1], angle_to_pin)

        # generate image
        state_img = np.zeros((self.STATE_IMAGE_HEIGHT, self.STATE_IMAGE_WIDTH, 3), np.uint8)
        state_img_y = 0

        for y in range(self.STATE_IMAGE_OFFSET_HEIGHT, self.STATE_IMAGE_HEIGHT + self.STATE_IMAGE_OFFSET_HEIGHT):
            state_img_x = 0
            for x in range(int(-self.STATE_IMAGE_WIDTH / 2), int(self.STATE_IMAGE_WIDTH / 2)):
                p1 = np.array([[y, x, 1]])
                p0 = np.dot(t01, p1.transpose())
                x0 = p0[0, 0]
                y0 = p0[1, 0]
                self.__test_x.append(x0)
                self.__test_y.append(y0)

                state_img[self.STATE_IMAGE_HEIGHT - state_img_y - 1, self.STATE_IMAGE_WIDTH - state_img_x - 1] = \
                    self.img[self.IMG_SIZE_Y - int(round(y0)), int(round(x0))]
                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1
        plt.imshow(state_img)
        plt.show()
