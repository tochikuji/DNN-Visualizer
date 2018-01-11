import chainer.functions as F


def deconvolution2d(node, convolved, raw):
    h = F.deconvolution_2d(convolved, node.W, stride=node.stride,
                           pad=node.pad, outsize=raw.shape[-2:])
    return h
