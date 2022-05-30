import math
from enum import IntEnum
import matplotlib.pyplot as plt
import numpy as np
from . import util
import cv2
from scipy.interpolate import interp1d
import os
import xml.etree.ElementTree as elemTree

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GolfEnv:
    class NoAreaInfoAssignedException(Exception):
        def __init__(self, pixel):
            self.pixel = pixel

        def __str__(self):
            return 'Cannot convert given pixel intensity ' + str(self.pixel) + ' to area info.'

    class InvalidInitialPosException(Exception):
        def __init__(self, pos, why):
            self.pos = pos
            self.why = why

        def __str__(self):
            return 'Cannot set initial pos ' + str(self.pos) + '. (' + self.why + ')'

    class MapConfigParseException(Exception):
        def __init__(self, path):
            self.path = path

        def __str__(self):
            return 'Error parsing config. File' + str(self.path) + ' seems to be corrupted.'

    class State:
        def __init__(self):
            self.distance_to_pin = None
            self.state_img = None
            self.ball_pos = None
            self.distance_to_pin = None
            self.landed_pixel_intensity = None
            self.club_availability = None

    class AreaInfoIndex(IntEnum):
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

    class ClubInfoIndex(IntEnum):
        NAME = 0
        DIST = 1
        DEV_X = 2
        DEV_Y = 3
        IS_DIST_PROPER = 4

    IMG_PATH_GRAY = os.path.join(os.path.dirname(__file__), '../resources/sophia_green_gray.png')
    IMG_PATH_COLOR = os.path.join(os.path.dirname(__file__), '../resources/sophia_green_gray.png')
    IMG_SIZE = np.array([500, 500])
    IMG_SAMPLING_STRIDE = 1 * 3.571
    START_POS = np.array([256, 116])
    PIN_POS = np.array([280, 430])
    STATE_IMAGE_WIDTH = 84
    STATE_IMAGE_HEIGHT = 84
    STATE_IMAGE_OFFSET_HEIGHT = -20 / 3.571
    OUT_OF_IMG_INTENSITY = 0

    # temporally disable Pycharm formatter for better readability
    # @formatter:off

    AREA_INFO = {
        # PIXL  NAME        K_DIST  K_DEV   ON_LAND                 TERM    RWRD(d: dist to pin)
        -1:     ('TEE',     1.0,    1.0,    OnLandAction.NONE,      False,  lambda d: -1),
        70:     ('FAIRWAY', 1.0,    1.0,    OnLandAction.NONE,      False,  lambda d: -1),
        80:     ('GREEN',   1.0,    1.0,    OnLandAction.NONE,      True,   lambda d: -1 + interp1d([0, 1, 3, 15, 100], [-1, -1, -2, -3, -3])(d)),
        50:     ('SAND',    0.6,    1.5,    OnLandAction.NONE,      False,  lambda d: -1),
        5:      ('WATER',   0.4,    1.0,    OnLandAction.SHORE,     False,  lambda d: -2),
        55:     ('ROUGH',   0.8,    1.5,    OnLandAction.NONE,      False,  lambda d: -1),
        0:      ('OB',      1.0,    1.0,    OnLandAction.ROLLBACK,  False,  lambda d: -3),
    }

    CLUB_INFO = (
        # NAME      DIST    DEV_X       DEV_Y       IS_DIST_PROPER(d: dist to pin)
        ('DR',      210.3,  54.8 / 3,   8.6 / 3,    lambda d: 300 < d),
        ('W3',      196.6,  50.3 / 3,   7.6 / 3,    lambda d: 100 < d),
        ('W5',      178.3,  36.6 / 3,   6.6 / 3,    lambda d: 100 < d),
        ('I3',      164.6,  36.6 / 3,   5.9 / 3,    lambda d: 100 < d),
        ('I4',      155.4,  32.0 / 3,   5.5 / 3,    lambda d: 100 < d),
        ('I5',      146.3,  27.4 / 3,   5.1 / 3,    lambda d: 100 < d <= 300),
        ('I6',      137.2,  27.4 / 3,   4.8 / 3,    lambda d: 100 < d <= 300),
        ('I7',      128.0,  27.4 / 3,   4.5 / 3,    lambda d: 100 < d <= 200),
        ('I8',      118.9,  27.4 / 3,   4.3 / 3,    lambda d: 100 < d <= 200),
        ('I9',      105.2,  32.0 / 3,   3.9 / 3,    lambda d: 100 < d <= 200),
        ('PW10',    96.0,   36.6 / 3,   3.7 / 3,    lambda d: 70 < d <= 200),
        ('SW9',     80,     36.6 / 3,   3.3 / 3,    lambda d: d <= 200),
        ('SW8',     70,     32.0 / 3,   3.2 / 3,    lambda d: d <= 200),
        ('SW7',     60,     30 / 3,     3.1 / 3,    lambda d: d <= 200),
        ('SW6',     50,     20 / 3,     3.0 / 3,    lambda d: d <= 200),
        ('SW5',     40,     15 / 3,     2.9 / 3,    lambda d: d <= 200),
        ('SW4',     30,     10 / 3,     2.8 / 3,    lambda d: d <= 200),
        ('SW3',     20,     5 / 3,      2.7 / 3,    lambda d: d <= 200),
        ('SW2',     10,     3 / 3,      2.7 / 3,    lambda d: d <= 200),
        ('SW1',     5,      1 / 3,      2.6 / 3,    lambda d: d <= 200),
    )

    # @formatter:on

    def __init__(self, map_name):
        # parse map config xml
        xml_path = os.path.join(os.path.dirname(__file__), '../configs', map_name+'.xml')
        tree = elemTree.parse(xml_path)

        try:
            self.IMG_PATH_GRAY = os.path.join(os.path.dirname(__file__), '..', tree.find('./img_path_gray').text)
            self.IMG_PATH_COLOR = os.path.join(os.path.dirname(__file__), '..', tree.find('./img_path_color').text)
            self.START_POS = np.array([
                int(tree.find('./tee/x').text),
                int(tree.find('./tee/y').text)
            ])
            self.PIN_POS = np.array([
                int(tree.find('./pin/x').text),
                int(tree.find('./pin/y').text)
            ])

        except AttributeError:
            raise self.MapConfigParseException(xml_path)

        self.__step_n = 0
        self.__max_step_n = -1
        self.__ball_path_x = []
        self.__ball_path_y = []
        self.__state = self.State()
        self.__img_color = cv2.cvtColor(cv2.imread(self.IMG_PATH_COLOR), cv2.COLOR_BGR2RGB)
        self.__img_gray = cv2.cvtColor(cv2.imread(self.IMG_PATH_GRAY), cv2.COLOR_BGR2GRAY)
        self.__rng = np.random.default_rng()

    def reset(self,
              initial_pos=None,
              randomize_initial_pos=False,
              max_timestep=-1,
              regenerate_club_availability=False
              ):
        """
        reset the environment
        :param initial_pos:
        :param randomize_initial_pos: randomly select initial position on green and rough
        :param max_timestep: terminates when step_n exceeds max_timestep
        :param regenerate_club_availability: randomize club availability.
        :return: tuple of initial state(img, dist), r:rewards term:termination
        """

        self.__max_step_n = max_timestep
        self.__step_n = 0
        self.__state.ball_pos = self.START_POS
        self.__state.club_availability = np.ones(len(GolfEnv.CLUB_INFO))
        self.__state.area_info = GolfEnv.AREA_INFO[self.__get_pixel_on(self.START_POS)]

        # randomize available clubs when club_availability is True
        if regenerate_club_availability:
            while True:
                self.__state.club_availability = np.random.randint(2, size=len(GolfEnv.CLUB_INFO))
                if np.max(self.__state.club_availability) == 1:
                    break

        # set initial pose when initial_pos is not None
        if initial_pos is not None:
            pixel = self.__get_pixel_on(initial_pos)

            if pixel not in GolfEnv.AREA_INFO:
                raise GolfEnv.NoAreaInfoAssignedException(pixel)

            area_info = GolfEnv.AREA_INFO[pixel]
            area_name = area_info[self.AreaInfoIndex.NAME]
            if not (
                    area_name == 'FAIRWAY' or
                    area_name == 'ROUGH'
            ):
                raise GolfEnv.InvalidInitialPosException(initial_pos, area_name)

            self.__state.area_info = area_info
            self.__state.ball_pos = initial_pos

        # randomize initial pose when randomize_initial_pos is True
        if randomize_initial_pos:
            while True:
                rand_pos = np.random.randint([0, 0], self.IMG_SIZE)
                pixel = self.__get_pixel_on(rand_pos)

                if pixel not in GolfEnv.AREA_INFO:
                    raise GolfEnv.NoAreaInfoAssignedException(pixel)

                area_info = GolfEnv.AREA_INFO[pixel]
                area_name = area_info[self.AreaInfoIndex.NAME]
                if(
                        area_name == 'FAIRWAY' or
                        area_name == 'ROUGH'
                ):
                    break

            self.__state.area_info = area_info
            self.__state.ball_pos = rand_pos

        # get ball pos, dist_to_pin
        self.__state.distance_to_pin = np.linalg.norm(self.__state.ball_pos - self.PIN_POS)
        self.__state.state_img = self.__generate_state_img(self.__state.ball_pos)
        self.__state.landed_pixel_intensity = self.__get_pixel_on(self.__state.ball_pos)

        self.__ball_path_x = [self.__state.ball_pos[0]]
        self.__ball_path_y = [self.__state.ball_pos[1]]

        return self.__state.state_img, self.__state.distance_to_pin, self.__state.club_availability

    def step(self, action, regenerate_heuristic_club_availability=False, accurate_shots=False, debug=False):
        """
        steps simulator
        :param regenerate_heuristic_club_availability:
        :param accurate_shots:
        :param action: tuple of action(continuous angle(deg), continuous distance(m))
        :param debug: print debug message of where the ball landed etc.
        :return: tuple of transition (s,r,term)
        s:tuple of state(img, dist), r:rewards term:termination
        """

        self.__step_n += 1

        debug_club_name = GolfEnv.CLUB_INFO[action[1]][GolfEnv.ClubInfoIndex.NAME]
        debug_area_name = ''

        if regenerate_heuristic_club_availability:
            self.__state.club_availability = self.__get_dist_proper_club_availability(self.__state.distance_to_pin)

        # when unavailable club is picked return previous state with reward of -4
        if self.__state.club_availability[action[1]] == 0:
            reward = -4
            # terminate when max step limit is reached
            termination = 0 < self.__max_step_n <= self.__step_n
            debug_club_name += ' (X)'

        else:
            # get area info, dist_coef, dev_coef
            self.__state.area_info = GolfEnv.AREA_INFO[self.__state.landed_pixel_intensity]
            dist_coef = self.__state.area_info[self.AreaInfoIndex.DIST_COEF]
            dev_coef = math.sqrt(self.__state.area_info[self.AreaInfoIndex.DEV_COEF])

            # get club info, distance, devs, reduced_dist
            self.__state.club_info = GolfEnv.CLUB_INFO[action[1]]
            club_name, club_distance, dev_x, dev_y, _ = self.__state.club_info
            reduced_dist = club_distance * dist_coef

            # nullify deviations if accurate_shots option is on
            if accurate_shots:
                dev_coef = 0.0

            # get tf delta of (x,y)
            angle_to_pin = math.atan2(self.PIN_POS[1] - self.__state.ball_pos[1],
                                      self.PIN_POS[0] - self.__state.ball_pos[0])
            shoot = np.array([[reduced_dist, 0]]) + self.__rng.normal(size=2,
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
            debug_area_name = area_info[GolfEnv.AreaInfoIndex.NAME]

            # get distance to ball
            distance_to_pin = np.linalg.norm(new_ball_pos - np.array([self.PIN_POS[0], self.PIN_POS[1]]))

            # get reward, termination from reward dict
            reward = area_info[self.AreaInfoIndex.REWARD](distance_to_pin)
            termination = area_info[self.AreaInfoIndex.TERMINATION]

            if area_info[self.AreaInfoIndex.ON_LAND] == self.OnLandAction.NONE:
                # get state img
                new_state_img = self.__generate_state_img(new_ball_pos)
                # update state
                self.__state.area_info = area_info
                self.__state.state_img = new_state_img
                self.__state.distance_to_pin = distance_to_pin
                self.__state.ball_pos = new_ball_pos
                self.__state.landed_pixel_intensity = new_pixel

            elif area_info[self.AreaInfoIndex.ON_LAND] == self.OnLandAction.ROLLBACK:
                # use previous state_img
                new_state_img = self.__state.state_img
                # add previous position to scatter plot to indicate ball return when rolled back
                self.__ball_path_x.append(self.__state.ball_pos[0])
                self.__ball_path_y.append(self.__state.ball_pos[1])

            elif self.__state.area_info[self.AreaInfoIndex.ON_LAND] == self.OnLandAction.SHORE:
                # get angle to move
                from_pin_vector = np.array([new_ball_pos[0] - self.PIN_POS[0], new_ball_pos[1] - self.PIN_POS[1]]).astype(
                    'float64')
                from_pin_vector /= np.linalg.norm(from_pin_vector)

                while True:
                    new_ball_pos += from_pin_vector
                    if not GolfEnv.AREA_INFO[self.__get_pixel_on(new_ball_pos)][
                               self.AreaInfoIndex.ON_LAND] == self.OnLandAction.SHORE:
                        break

                # get state img
                new_state_img = self.__generate_state_img(new_ball_pos)
                # recompute area info
                area_info = GolfEnv.AREA_INFO[self.__get_pixel_on(new_ball_pos)]
                # update state
                self.__state.area_info = area_info
                self.__state.state_img = new_state_img
                self.__state.distance_to_pin = distance_to_pin
                self.__state.ball_pos = new_ball_pos
                self.__state.landed_pixel_intensity = new_pixel

                # add current point to scatter plot to indicate on-landing action
                self.__ball_path_x.append(new_ball_pos[0])
                self.__ball_path_y.append(new_ball_pos[1])

        # print debug
        if debug:
            print(
                f'{self.__step_n:<7}'
                f'{debug_club_name:<10}'
                f'{self.__state.distance_to_pin:<6.2f}m    '
                f'{debug_area_name:<12}'
                f'reward:{reward:<6.2f}    '
                f'action:[{action[0]:<6.2f},{action[1]:<3}]'
            )

        if 0 < self.__max_step_n <= self.__step_n:
            termination = True

        return (self.__state.state_img, self.__state.distance_to_pin,
                self.__state.club_availability), reward, termination

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([0, self.IMG_SIZE[0]])
        plt.ylim([0, self.IMG_SIZE[1]])
        plt.imshow(plt.imread(self.IMG_PATH_GRAY), extent=[0, self.IMG_SIZE[0], 0, self.IMG_SIZE[1]])
        plt.plot(self.__ball_path_x, self.__ball_path_y, marker='o', color="white")

        plt.show()

    def paint(self):
        img = np.copy(self.__img_color)

        # draw dots
        for i in range(len(self.__ball_path_x)):
            img = cv2.circle(img, (int(self.__ball_path_x[i]), self.IMG_SIZE[1]-1 - int(self.__ball_path_y[i])), 3, (255,255,255), cv2.FILLED, cv2.LINE_8)
        # draw lines
        for i in range(len(self.__ball_path_x)-1):
            img = cv2.line(img,
                           (int(self.__ball_path_x[i]), self.IMG_SIZE[1] - 1 - int(self.__ball_path_y[i])),
                           (int(self.__ball_path_x[i+1]), self.IMG_SIZE[1] - 1 - int(self.__ball_path_y[i+1])),
                           (255, 255, 255),
                           1,
                           cv2.LINE_AA)
        # plt.imshow(img, extent=[0, self.IMG_SIZE[0], 0, self.IMG_SIZE[1]])
        # plt.show()
        return img

    def __get_dist_proper_club_availability(self, dist):
        club_n = len(GolfEnv.CLUB_INFO)
        availability = np.zeros(club_n)
        for i in range(club_n):
            availability[i] = int(GolfEnv.CLUB_INFO[i][GolfEnv.ClubInfoIndex.IS_DIST_PROPER](dist))
        return availability

    def __get_pixel_on(self, ball_pos):
        x0 = int(round(ball_pos[0]))
        y0 = int(round(ball_pos[1]))
        if util.is_within([0, 0], [self.IMG_SIZE[0] - 1, self.IMG_SIZE[1] - 1], [x0, y0]):
            return self.__img_gray[-y0 - 1, x0]
        else:
            return self.OUT_OF_IMG_INTENSITY

    def __generate_state_img(self, pos):
        # get angle
        angle_to_pin = math.atan2(self.PIN_POS[1] - pos[1], self.PIN_POS[0] - pos[0])

        # get tf between fixed frame and moving frame (to use p0 = t01*p1)
        t01 = util.transform_2d(pos[0], pos[1], angle_to_pin)

        # generate image
        state_img = np.zeros((self.STATE_IMAGE_HEIGHT, self.STATE_IMAGE_WIDTH), np.uint8)
        state_img_y = 0

        for y in range(int(self.STATE_IMAGE_OFFSET_HEIGHT), self.STATE_IMAGE_HEIGHT + int(self.STATE_IMAGE_OFFSET_HEIGHT)):
            state_img_x = 0
            for x in range(int(-self.STATE_IMAGE_WIDTH / 2), int(self.STATE_IMAGE_WIDTH / 2)):
                p1 = np.array([y*self.IMG_SAMPLING_STRIDE, x*self.IMG_SAMPLING_STRIDE, 1])
                p0 = np.dot(t01, p1)
                x0 = int(round(p0[0]))
                y0 = int(round(p0[1]))

                if util.is_within([0, 0], [self.IMG_SIZE[0] - 1, self.IMG_SIZE[1] - 1], [x0, y0]):
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.__img_gray[-y0 - 1, x0]
                else:
                    state_img[- state_img_y - 1, - state_img_x - 1] = self.OUT_OF_IMG_INTENSITY

                state_img_x = state_img_x + 1
            state_img_y = state_img_y + 1

        return state_img
