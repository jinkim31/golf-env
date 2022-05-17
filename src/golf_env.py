import math
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
import util
import cv2
from scipy.interpolate import interp1d
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GolfEnv:
    class NoAreaInfoAssignedException(Exception):
        def __init__(self, pixel):
            self.pixel = pixel

        def __str__(self):
            return 'Cannot convert given pixel intensity ' + str(self.pixel) + ' to area info.'

    class State:
        def __init__(self):
            self.distance_to_pin = None
            self.state_img = None
            self.ball_pos = None
            self.distance_to_pin = None
            self.landed_pixel_intensity = None
            self.flight_model = None

    class AreaInfo(IntEnum):
        NAME = 0
        DIST_COEF = 1
        DEV_COEF = 2
        ON_LAND = 3
        TERMINATION = 4
        REWARD = 5

    class OnLandAction(IntEnum):
        NONE = 0
        ROLLBACK = 1
        SHORE = 2

    IMG_PATH = "resources/env.png"
    IMG_SIZE = np.array([500, 500])
    START_POS = np.array([256, 116])
    PIN_POS = np.array([280, 430])
    STATE_IMAGE_WIDTH = 300
    STATE_IMAGE_HEIGHT = 300
    STATE_IMAGE_OFFSET_HEIGHT = -20
    OUT_OF_IMG_INTENSITY = 0
    AREA_INFO = {
        # PIXL   NAME       K_DIST  K_DEV   ON_LAND                     TERM    RWRD
        -1: ('TEE', 1.0, 1.0, OnLandAction.NONE, False, lambda d: -1),
        70: ('FAIRWAY', 1.0, 1.0, OnLandAction.NONE, False, lambda d: -1),
        80: ('GREEN', 1.0, 1.0, OnLandAction.NONE, True, lambda d: -1 + interp1d([0, 1, 3, 15, 100], [-1, -1, -2, -3, -3])(d)),
        50: ('SAND', 0.6, 1.5, OnLandAction.NONE, False, lambda d: -1),
        5: ('WATER', 0.4, 1.0, OnLandAction.SHORE, False, lambda d: -2),
        55: ('ROUGH', 0.8, 1.5, OnLandAction.NONE, False, lambda d: -1),
        0: ('OB', 1.0, 1.0, OnLandAction.ROLLBACK, False, lambda d: -3),
    }
    FLIGHT_MODELS = (
        # NAME  DIST    DEV_X   DEV_Y
        ('DR', 230, 60, 60),
        ('W3', 215, 55, 55),
        ('W5', 195, 40, 40),
        ('I3', 180, 40, 40),
        ('I4', 170, 35, 35),
        ('I5', 160, 30, 30),
        ('I6', 150, 30, 30),
        ('I7', 140, 30, 30),
        ('I8', 130, 30, 30),
        ('I9', 115, 35, 35),
        ('PW', 105, 40, 40),
        ('SW', 80, 40, 40),
        ('SW', 70, 35, 35),
        ('SW', 60, 30, 30),
        ('SW', 50, 20, 20),
        ('SW', 40, 15, 15),
        ('SW', 30, 10, 10),
        ('SW', 20, 5, 5),
        ('SW', 10, 3, 3),
        ('SW', 5, 1, 1),
    )

    def __init__(self):
        self.__step_n = 0
        self.__max_step_n = -1
        self.__ball_path_x = []
        self.__ball_path_y = []
        self.__state = self.State()
        self.__img = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2RGB)
        self.__img_gray = cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2GRAY)
        self.__rng = np.random.default_rng()

    def reset(self, randomize_initial_pos=False, max_timestep=-1):
        """
        :return: tuple of initial state(img, dist), r:rewards term:termination
        """

        self.__max_step_n = max_timestep
        self.__step_n = 0
        self.__ball_path_x = [self.START_POS[0]]
        self.__ball_path_y = [self.START_POS[1]]
        self.__state.ball_pos = self.START_POS

        if randomize_initial_pos:
            while True:
                rand_pos = np.random.randint([0, 0], self.IMG_SIZE)
                pixel = self.__get_pixel_on(rand_pos)

                if pixel not in self.__area_info:
                    raise GolfEnv.NoAreaInfoAssignedException(pixel)

                area_info = self.__area_info[pixel]
                if area_info[self.AreaInfo.NAME] == 'FAIRWAY' or area_info[self.AreaInfo.NAME] == 'ROUGH':
                    break

            self.__state.ball_pos = rand_pos

        # get ball pos, dist_to_pin
        self.__state.distance_to_pin = np.linalg.norm(self.__state.ball_pos - self.PIN_POS)
        self.__state.state_img = self.__generate_state_img(self.__state.ball_pos[0], self.__state.ball_pos[1])
        self.__state.landed_pixel_intensity = self.__get_pixel_on(self.__state.ball_pos)

        self.__ball_path_x = [self.__state.ball_pos[0]]
        self.__ball_path_y = [self.__state.ball_pos[1]]

        return self.__state.state_img, self.__state.distance_to_pin

    def _customize_debug_str(self, msg):
        """
        template method for customizing debug msg
        :param msg: original debug string
        :return: customized debug string
        """
        return msg

    def step(self, action, accurate_shots=False, debug=False):
        """
        steps simulator
        :param action: tuple of action(continuous angle(deg), continuous distance(m))
        :param debug: print debug message of where the ball landed etc.
        :return: tuple of transition (s,r,term)
        s:tuple of state(img, dist), r:rewards term:termination
        """
        self.__step_n += 1

        # get flight model
        area_info = GolfEnv.AREA_INFO[self.__state.landed_pixel_intensity]
        dist_coef = area_info[self.AreaInfo.DIST_COEF]
        dev_coef = math.sqrt(area_info[self.AreaInfo.DEV_COEF])
        self.__state.flight_model = GolfEnv.FLIGHT_MODELS[action[1]]
        club_name, distance, dev_x, dev_y = self.__state.flight_model
        reduced_distance = distance * dist_coef

        # nullify deviations if accurate_shots option is on
        if accurate_shots:
            dev_coef = 0.0

        # get tf delta of (x,y)
        angle_to_pin = math.atan2(self.PIN_POS[1] - self.__state.ball_pos[1],
                                  self.PIN_POS[0] - self.__state.ball_pos[0])
        shoot = np.array([[reduced_distance, 0]]) + self.__rng.normal(size=2,
                                                                      scale=[dev_x * dev_coef, dev_y * dev_coef])
        delta = np.dot(util.rotation_2d(util.deg_to_rad(action[0]) + angle_to_pin), shoot.transpose()).transpose()

        # offset tf by delta to derive new ball pose
        new_ball_pos = np.array([self.__state.ball_pos[0] + delta[0][0], self.__state.ball_pos[1] + delta[0][1]])

        # store position for plotting
        self.__ball_path_x.append(new_ball_pos[0])
        self.__ball_path_y.append(new_ball_pos[1])

        # get landed pixel intensity, area info
        new_pixel = self.__get_pixel_on(new_ball_pos)
        if new_pixel not in GolfEnv.AREA_INFO:
            raise GolfEnv.NoAreaInfoAssignedException(new_pixel)
        area_info = GolfEnv.AREA_INFO[new_pixel]

        # get distance to ball
        distance_to_pin = np.linalg.norm(new_ball_pos - np.array([self.PIN_POS[0], self.PIN_POS[1]]))

        # get reward, termination from reward dict
        reward = area_info[self.AreaInfo.REWARD](distance_to_pin)
        termination = area_info[self.AreaInfo.TERMINATION]

        if area_info[self.AreaInfo.ON_LAND] == self.OnLandAction.NONE:
            # get state img
            state_img = self.__generate_state_img(new_ball_pos[0], new_ball_pos[1])

            # update state
            self.__state.state_img = state_img
            self.__state.distance_to_pin = distance_to_pin
            self.__state.ball_pos = new_ball_pos
            self.__state.landed_pixel_intensity = new_pixel

        elif area_info[self.AreaInfo.ON_LAND] == self.OnLandAction.ROLLBACK:
            # add previous position to scatter plot to indicate ball return when rolled back
            self.__ball_path_x.append(self.__state.ball_pos[0])
            self.__ball_path_y.append(self.__state.ball_pos[1])

        elif area_info[self.AreaInfo.ON_LAND] == self.OnLandAction.SHORE:
            # get angle to move
            from_pin_vector = np.array([new_ball_pos[0] - self.PIN_POS[0], new_ball_pos[1] - self.PIN_POS[1]]).astype(
                'float64')
            from_pin_vector /= np.linalg.norm(from_pin_vector)

            while True:
                new_ball_pos += from_pin_vector
                if not self.__area_info[self.__get_pixel_on(new_ball_pos)][self.AreaInfo.ON_LAND] == self.OnLandAction.SHORE:
                    break

            # get state img
            state_img = self.__generate_state_img(new_ball_pos[0], new_ball_pos[1])

            # add current point to scatter plot to indicate on-landing action
            self.__ball_path_x.append(new_ball_pos[0])
            self.__ball_path_y.append(new_ball_pos[1])

            # update state
            self.__state.state_img = state_img
            self.__state.distance_to_pin = distance_to_pin
            self.__state.ball_pos = new_ball_pos
            self.__state.landed_pixel_intensity = new_pixel

        # print debug
        if debug:
            print('itr' + str(self.__step_n) + ': ' + self._customize_debug_str(
                'landed on ' + area_info[self.AreaInfo.NAME] +
                ' dist_coef:' + str(dist_coef) +
                ' dev_coef:' + str(dev_coef) +
                ' on_land:' + str(area_info[self.AreaInfo.ON_LAND]) +
                ' termination:' + str(termination) +
                ' distance:' + str(self.__state.distance_to_pin) +
                ' reward:' + str(reward)))

        if 0 < self.__max_step_n <= self.__step_n:
            termination = True

        return (self.__state.state_img, self.__state.distance_to_pin), reward, termination

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, self.IMG_SIZE[0]])
        plt.ylim([0, self.IMG_SIZE[1]])
        plt.imshow(plt.imread(self.IMG_PATH), extent=[0, self.IMG_SIZE[0], 0, self.IMG_SIZE[1]])
        plt.plot(self.__ball_path_x, self.__ball_path_y, marker='o', color="white")
        plt.show()

    def __get_pixel_on(self, ball_pos):
        x0 = int(round(ball_pos[0]))
        y0 = int(round(ball_pos[1]))
        if util.is_within([0, 0], [self.IMG_SIZE[0] - 1, self.IMG_SIZE[1] - 1], [x0, y0]):
            return self.__img_gray[-y0 - 1, x0]
        else:
            return self.OUT_OF_IMG_INTENSITY

    def __generate_state_img(self, x, y):
        # get angle
        angle_to_pin = math.atan2(self.PIN_POS[1] - y, self.PIN_POS[0] - x)

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

                if util.is_within([0, 0], [self.IMG_SIZE[0] - 1, self.IMG_SIZE[1] - 1], [x0, y0]):
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.__img_gray[-y0 - 1, x0]
                else:
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.OUT_OF_IMG_INTENSITY

                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1

        return state_img
