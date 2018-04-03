# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

from IPython import embed
import tensorflow as tf
import numpy as np

def _debug_single(x):
    print(x.shape)
    np.save('/tmp/x', x)
    embed()
    return True


def _debug_two(x, y):
    embed()
    return True


def _debug_three(x, y, z):
    embed()
    return True


def _debug_four(x, y, z, u):
    embed()
    return True
