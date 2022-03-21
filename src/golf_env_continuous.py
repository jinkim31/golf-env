import math
from enum import IntEnum
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
    VAR_X = 0
    VAR_Y = 0
    STATE_IMAGE_WIDTH = 300
    STATE_IMAGE_HEIGHT = 300
    STATE_IMAGE_OFFSET_HEIGHT = -20
    OUT_OF_IMG_INTENSITY = 255

    class NoAreaInfoAssignedException(Exception):
        def __init__(self, pixel):
            self.pixel = pixel

        def __str__(self):
            return 'Cannot convert given pixel intensity ' + str(self.pixel) + ' to area info.'

    class AreaInfo(IntEnum):
        NAME = 0
        REDUCTION = 1
        ROLLBACK = 2
        TERMINATION = 3
        REWARD = 4

    def __init__(self):
        self.__step_n = 0
        self.__state = None
        self.__arrow_head_x = []
        self.__arrow_head_y = []
        self.__arrow_tail_x = []
        self.__arrow_tail_y = []
        self.__state_img_mask_x = []
        self.__state_img_mask_y = []
        self.__ball_path_x = []
        self.__ball_path_y = []
        self.__ball_pos = None
        self.__distance_to_pin = 0
        self.__prev_pixel = 0
        self.__img = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2RGB)
        self.__img_gray = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2GRAY)
        self.__area_info = {
            # PIXL   NAME       RDUX    RBCK    TERM    RWRD
            -1:     ('TEE',     1.0,    False,  False,  lambda: -1),
            160:    ('FAREWAY', 1.0,    False,  False,  lambda: -1),
            83:     ('GREEN',   1.0,    False,  True,   lambda: -1 + self.green_reward_function(self.__distance_to_pin)),
            231:    ('SAND',    0.4,    False,  False,  lambda: -1),
            -1:     ('WATER',   0.4,    False,  False,  lambda: -1),
            77:     ('ROUGH',   1.0,    False,  False,  lambda: -1),
            0:      ('OB',      1.0,    True,   False,  lambda: -2),
            255:    ('OB',      1.0,    True,   False,  lambda: -2)
        }
        self.green_reward_function = interp1d(np.array([0, 1, 3, 15, 100]), np.array([-1, -1, -2, -3, -3]))
        self.rng = np.random.default_rng()

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
        reduction = self.__area_info[self.__prev_pixel][self.AreaInfo.REDUCTION]
        reduced_distance = action[1] * reduction
        angle_to_pin = math.atan2(self.PIN_Y - self.__ball_pos[1], self.PIN_X - self.__ball_pos[0])
        shoot = np.array([[reduced_distance, 0]]) + self.rng.normal(size=2, scale=[self.VAR_X, self.VAR_Y])
        delta = np.dot(util.rotation_2d(action[0] + angle_to_pin), shoot.transpose()).transpose()

        # offset tf by delta to derive new ball pose
        new_ball_pos = np.array([self.__ball_pos[0] + delta[0][0], self.__ball_pos[1] + delta[0][1]])

        # store position for plotting
        self.__ball_path_x.append(new_ball_pos[0])
        self.__ball_path_y.append(new_ball_pos[1])

        # get landed pixel intensity, area info
        new_pixel = self.__get_pixel_on(new_ball_pos)
        if new_pixel not in self.__area_info:
            raise GolfEnv.NoAreaInfoAssignedException(new_pixel)
        area_info = self.__area_info[new_pixel]

        # get reward, termination from reward dict
        reward = area_info[self.AreaInfo.REWARD]()
        termination = area_info[self.AreaInfo.TERMINATION]

        if not area_info[self.AreaInfo.ROLLBACK]:
            # get state img
            state_img, p = self.__generate_state_img(new_ball_pos[0], new_ball_pos[1])

            # get distance to ball
            self.__distance_to_pin = np.linalg.norm(new_ball_pos - np.array([self.PIN_X, self.PIN_Y]))

            # update state
            self.__state = (state_img, self.__distance_to_pin)
            self.__ball_pos = new_ball_pos
            self.__prev_pixel = new_pixel
        else:
            # add previous position to indicate ball return when rolled back
            self.__ball_path_x.append(self.__ball_pos[0])
            self.__ball_path_y.append(self.__ball_pos[1])

        # print debug
        if debug:
            print(
                'itr' + str(self.__step_n) +
                ': landed on ' + area_info[self.AreaInfo.NAME] +
                ' reduction:' + str(reduction) +
                ' reward:' + str(reward) +
                ' rollback:' + str(area_info[self.AreaInfo.ROLLBACK]) +
                ' termination:' + str(termination))

        return self.__state, reward, termination

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, self.IMG_SIZE_X])
        plt.ylim([0, self.IMG_SIZE_Y])
        plt.imshow(plt.imread(self.IMG_PATH), extent=[0, self.IMG_SIZE_X, 0, self.IMG_SIZE_Y])
        # plt.scatter(self.PIN_X, self.PIN_Y, s=500, marker='x', color='black')
        # plt.scatter(self.START_X, self.START_Y, s=200, color='black')
        # plt.quiver(self.__arrow_head_x, self.__arrow_head_y, self.__arrow_tail_x, self.__arrow_tail_y)
        plt.plot(self.__ball_path_x, self.__ball_path_y, marker='o', color="black")
        # plt.scatter(self.__state_img_mask_x, self.__state_img_mask_y, s=0.01, color='black')
        plt.show()

    def reset(self):
        """
        :return: tuple of initial state(img, dist), r:rewards term:termination
        """
        self.__step_n = 0
        self.__arrow_head_x = []
        self.__arrow_head_y = []
        self.__arrow_tail_x = []
        self.__arrow_tail_y = []
        self.__state_img_mask_x = []
        self.__state_img_mask_y = []
        self.__ball_path_x = [self.START_X]
        self.__ball_path_y = [self.START_Y]

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

    def __get_pixel_on(self, ball_pos):
        x0 = int(round(ball_pos[0]))
        y0 = int(round(ball_pos[1]))
        if util.is_within([0, 0], [self.IMG_SIZE_X - 1, self.IMG_SIZE_Y - 1], [x0, y0]):
            return self.__img_gray[-y0 - 1, x0]
        else:
            return self.OUT_OF_IMG_INTENSITY

    def __generate_state_img(self, x, y):
        # save data to plot
        angle_to_pin = math.atan2(self.PIN_Y - y, self.PIN_X - x)
        arrow = np.dot(util.rotation_2d(angle_to_pin), np.array([[1, 0]]).transpose())
        self.__arrow_head_x.append(x)
        self.__arrow_head_y.append(y)
        self.__arrow_tail_x.append(arrow[0, 0])
        self.__arrow_tail_y.append(arrow[1, 0])

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
                self.__state_img_mask_x.append(x0)
                self.__state_img_mask_y.append(y0)

                if util.is_within([0, 0], [self.IMG_SIZE_X - 1, self.IMG_SIZE_Y - 1], [x0, y0]):
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.__img_gray[-y0 - 1, x0]
                else:
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.OUT_OF_IMG_INTENSITY

                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1

        # get pixel intensity from generated state image
        landed_pixel_intensity = state_img[int(self.STATE_IMAGE_OFFSET_HEIGHT - 1), int(self.STATE_IMAGE_WIDTH / 2)]

        return state_img, landed_pixel_intensity
