import numpy
import numpy as np
import math


def deg_to_rad(deg):
    return deg / 180.0 * math.pi


def rotation_2d(rot):
    return np.array([[math.cos(rot), -math.sin(rot)],
                     [math.sin(rot), math.cos(rot)]])


def transform_2d(tr_x, tr_y, rot):
    return np.array([[math.cos(rot), -math.sin(rot), tr_x],
                     [math.sin(rot), math.cos(rot), tr_y],
                     [0, 0, 1]])


def inv_transform_2d(tf):
    assert (tf.shape == (3, 3))

    rot = tf[:2, :2]
    translation = tf[:2, 2:3]
    inv = np.concatenate([rot.transpose(), np.dot(-rot.transpose(), translation)], 1)
    inv = numpy.concatenate([inv, numpy.array([[0, 0, 1]])])
    return inv
