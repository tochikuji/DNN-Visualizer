import numpy


def inverse_linear(node, transformed, raw):
    vec = transformed.reshape(-1, 1)

    ret = numpy.dot(node.W.data.T, vec)
    ret -= node.b.data.reshape(-1, 1)

    return ret.reshape(raw.shape)