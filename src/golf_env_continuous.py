import math
import matplotlib.pyplot as plt
import numpy as np
import util
import env
import cv2


class GolfEnv(env.Env):
    IMG_PATH = "resources/img.png"
    IMG_SIZE_X = 100
    IMG_SIZE_Y = 100
    START_X = 10
    START_Y = 10
    PIN_X = 50
    PIN_Y = 90
    VAR_X = 1
    VAR_Y = 3
    STATE_IMAGE_WIDTH = 20
    STATE_IMAGE_HEIGHT = 30
    STATE_IMAGE_OFFSET_HEIGHT = -4

    def __init__(self):
        super().__init__(2, 2)  # action size:2(angle,club), state size:2(x,y)
        self.__state = None
        self.__marker_x = []
        self.__marker_y = []
        self.__marker_end_x = []
        self.__marker_end_y = []
        self.__test_x = []
        self.__test_y = []
        self.img = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2GRAY)

    def step(self, action):
        """
        steps simulator
        :param action: tuple of action (angle, distance) angle:continuous distance:continuous
        :return: tuple of transition (s,a,r,s')
        s:before state(x,y,img,t) a:action, r:reward, s':new state(x',y',img,t)
        """

        rng = np.random.default_rng()

        angle_to_pin = math.atan2(self.PIN_Y - self.__state[1], self.PIN_X - self.__state[0])
        shoot = np.array([[action[1], 0]]) + rng.normal(size=2, scale=[self.VAR_X, self.VAR_Y])
        delta = np.dot(util.rotation_2d(action[0] + angle_to_pin), shoot.transpose()).transpose()

        new_x = self.__state[0] + delta[0][0]
        new_y = self.__state[1] + delta[0][1]

        old_state = self.__state
        state_img = self.__generate_state_img(new_x, new_y)
        new_state = (new_x, new_y, state_img, False)
        self.__state = new_state

        reward = 0

        return old_state, action, reward, new_state

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, self.IMG_SIZE_X])
        plt.ylim([0, self.IMG_SIZE_Y])
        plt.imshow(plt.imread(self.IMG_PATH), extent=[0, self.IMG_SIZE_X, 0, self.IMG_SIZE_Y])
        plt.scatter(self.PIN_X, self.PIN_Y, s=500, marker='x', color='black')
        plt.scatter(self.START_X, self.START_Y, s=200, color='black')
        plt.quiver(self.__marker_x, self.__marker_y, self.__marker_end_x, self.__marker_end_y)
        plt.scatter(self.__test_x, self.__test_y, s=0.1, color='black')
        plt.show()

    def reset(self):
        """
        :return: tuple of initial state (x, y, img, t) t:termination
        """
        self.__marker_x = []
        self.__marker_y = []
        self.__marker_end_x = []
        self.__marker_end_y = []
        self.__test_x = []
        self.__test_y = []

        self.__state = (self.START_X, self.START_Y, self.__generate_state_img(self.START_X, self.START_Y), False)
        return self.__state

    def __generate_state_img(self, x, y):
        # save data to plot
        angle_to_pin = math.atan2(self.PIN_Y - y, self.PIN_X - x)
        arrow = np.dot(util.rotation_2d(angle_to_pin), np.array([[1, 0]]).transpose())
        self.__marker_x.append(x)
        self.__marker_y.append(y)
        self.__marker_end_x.append(arrow[0, 0])
        self.__marker_end_y.append(arrow[1, 0])

        # get tf between fixed frame and moving frame (to use p0 = t01*p1)
        t01 = util.transform_2d(x, y, angle_to_pin)

        # generate image
        state_img = np.zeros((self.STATE_IMAGE_HEIGHT, self.STATE_IMAGE_WIDTH), np.uint8)
        state_img_y = 0

        for y in range(self.STATE_IMAGE_OFFSET_HEIGHT, self.STATE_IMAGE_HEIGHT + self.STATE_IMAGE_OFFSET_HEIGHT):
            state_img_x = 0
            for x in range(int(-self.STATE_IMAGE_WIDTH / 2), int(self.STATE_IMAGE_WIDTH / 2)):
                p1 = np.array([[y, x, 1]])
                p0 = np.dot(t01, p1.transpose())
                x0 = int(round(p0[0, 0]))
                y0 = int(round(p0[1, 0]))
                self.__test_x.append(x0)
                self.__test_y.append(y0)

                if util.is_within([0, 0], [self.IMG_SIZE_X - 1, self.IMG_SIZE_Y - 1], [x0, y0]):
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.img_gray[-y0 - 1, x0]
                else:
                    state_img[- state_img_y - 1, - state_img_x - 1] = 0

                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1

        return state_img
