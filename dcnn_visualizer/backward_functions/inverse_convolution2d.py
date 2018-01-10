import chainer.functions as F


def deconvolution2d(node, convolved, raw):
    h = F.deconvolution_2d(convolved, node.W, node.b, stride=node.stride, pad=node.pad, outsize=raw.shape)
    return h
