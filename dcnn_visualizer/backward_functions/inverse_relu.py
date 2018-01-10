import chainer.functions as F


def inverse_relu_anew(node, rectified, raw):
    h = F.relu(rectified)

    return h


def inverse_relu_locational(node, rectified, raw):
    positives = rectified * (raw > 0)

    return positives


def relu(x):
    return x * (x > 0)
