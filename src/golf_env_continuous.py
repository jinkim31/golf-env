import math
import matplotlib.pyplot as plt
import numpy as np
import util
import cv2
from scipy.interpolate import interp1d


class GolfEnv:
    IMG_PATH = "resources/env.png"
    IMG_SIZE_X = 500
    IMG_SIZE_Y = 500
    START_X = 260
    START_Y = 120
    PIN_X = 280
    PIN_Y = 430
    VAR_X = 10
    VAR_Y = 10
    STATE_IMAGE_WIDTH = 300
    STATE_IMAGE_HEIGHT = 300
    STATE_IMAGE_OFFSET_HEIGHT = -20
    OUT_OF_IMG_INTENSITY = 255

    class NoAreaNameAssignedException(Exception):
        def __init__(self, pixel):
            self.pixel = pixel

        def __str__(self):
            return 'Cannot convert given pixel intensity ' + str(self.pixel) + ' to area name.'

    def __init__(self):
        self.__step_n = 0
        self.__state = None
        self.__marker_x = []
        self.__marker_y = []
        self.__marker_end_x = []
        self.__marker_end_y = []
        self.__test_x = []
        self.__test_y = []
        self.__ball_pos = None
        self.__distance_to_pin = 0
        self.__prev_pixel = 0
        self.__img = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2RGB)
        self.__img_gray = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2GRAY)
        self.__area_info = {
            -1: ('TEE', 1.0, False, lambda: -1),
            160: ('FAREWAY', 1.0, False, lambda: -1),
            83: ('GREEN', 1.0, True, lambda: -1 + self.get_green_reward(self.__distance_to_pin)),
            231: ('SAND', 0.4, False, lambda: -1),
            -1: ('WATER', 0.4, False, lambda: -1),
            77: ('ROUGH', 1.0, False, lambda: -1),
            0: ('OB', 1.0, True, lambda: -100),
            255: ('OB', 1.0, True, lambda: -100)
        }

    def step(self, action, debug=False):
        """
        steps simulator
        :param action: tuple of action(continuous angle, continuous distance)
        :param debug: print debug message of where the ball landed etc.
        :return: tuple of transition (s,r,term)
        s:tuple of state(img, dist), r:rewards term:termination
        """
        self.__step_n += 1

        # get tf delta of (x,y)
        rng = np.random.default_rng()
        reduced_distance = action[1] * self.__area_info[self.__prev_pixel][1]
        angle_to_pin = math.atan2(self.PIN_Y - self.__ball_pos[1], self.PIN_X - self.__ball_pos[0])
        shoot = np.array([[reduced_distance, 0]]) + rng.normal(size=2, scale=[self.VAR_X, self.VAR_Y])
        delta = np.dot(util.rotation_2d(action[0] + angle_to_pin), shoot.transpose()).transpose()

        # offset tf by delta
        self.__ball_pos = np.array([self.__ball_pos[0] + delta[0][0], self.__ball_pos[1] + delta[0][1]])

        # get state img
        state_img, pixel = self.__generate_state_img(self.__ball_pos[0], self.__ball_pos[1])
        self.__prev_pixel = pixel

        # get distance to ball
        self.__distance_to_pin = np.linalg.norm(self.__ball_pos - np.array([self.PIN_X, self.PIN_Y]))

        # get reward, termination from reward dict
        if pixel not in self.__area_info:
            raise GolfEnv.NoAreaNameAssignedException(pixel)
        reward = self.__area_info[pixel][3]()
        termination = self.__area_info[pixel][2]

        # update state
        self.__state = (state_img, self.__distance_to_pin)

        # print debug
        if debug:
            print('itr' + str(self.__step_n) + ': landed on ' + self.__area_info[pixel][0] + ' reward:' + str(reward) +
                  ' termination:'+str(termination))

        return self.__state, reward, termination

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
        plt.scatter(self.__test_x, self.__test_y, s=0.01, color='black')
        plt.show()

    def reset(self):
        """
        :return: tuple of initial state(img, dist), r:rewards term:termination
        """
        self.__step_n = 0
        self.__marker_x = []
        self.__marker_y = []
        self.__marker_end_x = []
        self.__marker_end_y = []
        self.__test_x = []
        self.__test_y = []

        # get ball pos, dist_to_pin
        self.__ball_pos = np.array([self.START_X, self.START_Y])
        dist_to_pin = np.linalg.norm(self.__ball_pos - np.array([self.PIN_X, self.PIN_Y]))

        state_img, pixel = self.__generate_state_img(self.START_X, self.START_Y)
        self.__prev_pixel = pixel
        self.__state = (state_img, dist_to_pin)
        return self.__state

    def show_grayscale(self):
        plt.imshow(cv2.cvtColor(self.__img_gray, cv2.COLOR_GRAY2BGR))
        plt.show()

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
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.__img_gray[-y0 - 1, x0]
                else:
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.OUT_OF_IMG_INTENSITY

                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1

        # get pixel intensity from generated state image
        landed_pixel_intensity = state_img[int(self.STATE_IMAGE_OFFSET_HEIGHT - 1), int(self.STATE_IMAGE_WIDTH / 2)]

        return state_img, landed_pixel_intensity

    def get_green_reward(self, distance_to_pin):
        x = np.array([0, 1, 3, 15])
        y = np.array([1, 1, 2, 3])
        f = interp1d(x, y)
        # xnew = np.linspace(0, 15, num=41, endpoint=True)
        # plt.plot(x, y, 'o', xnew, f(xnew), '-')
        # plt.show()
        # print('distance:' + str(distance_to_pin) + ' reward:' + str(-f(distance_to_pin)))
        return -f(distance_to_pin)
