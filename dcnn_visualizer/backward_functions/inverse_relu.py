import numpy
import chainer.functions as F
from dcnn_visualizer.util import expect_ndarray


def inverse_relu_anew(node, rectified, raw):
    h = F.relu(rectified)

    return h


def inverse_relu_locational(node, rectified, raw):
    with expect_ndarray(rectified) as re_data:
        with expect_ndarray(raw) as raw_data:
            positives = re_data * (raw_data > 0)

    return positives


def relu(x):
    with expect_ndarray(x) as data:
        return data * (data > 0)
