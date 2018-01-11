import numpy
from dcnn_visualizer.util import expect_ndarray


def inverse_linear(node, transformed, raw):
    vec = transformed.reshape(-1, 1)

    with expect_ndarray(node.W) as weight, expect_ndarray(vec) as vec_arr, \
            expect_ndarray(node.b) as bias:
        ret = numpy.dot(weight.T, vec_arr - bias.reshape(-1, 1))

    return ret.reshape(raw.shape)