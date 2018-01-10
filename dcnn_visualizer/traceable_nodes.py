import chainer
import chainer.links as L
import chainer.functions as F


class TraceableNode:
    def __init__(self):
        self.traceable_node_type = None

    def __call__(self, *args, **opt):
        raise NotImplementedError()


class TraceableFunctionWrapper(TraceableNode):
    def __init__(self, *args, **opt):
        super().__init__()

        self._func = None

        self.args = args
        self.opt = opt

    def __call__(self, x):
        return self._func(x, *self.args, **self.opt)


class TraceableConvolution2D(L.Convolution2D, TraceableNode):
    def __init__(self, *args, **opt):
        super().__init__(*args, **opt)
        self.traceable_node_type = 'CONV'


class TraceableLinear(L.Linear, TraceableNode):
    def __init__(self, *args, **opt):
        super().__init__(*args, **opt)
        self.traceable_node_type = 'AFFINE'


class TraceableMaxPooling2D(TraceableFunctionWrapper):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True, *args, **opt):
        super().__init__(*args, **opt)

        self._func = F.max_pooling_2d
        self.traceable_node_type = 'MP'

        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all

    def __call__(self, x, *args, **opt):
        return self._func(x, ksize=self.ksize, stride=self.stride, pad=self.pad, cover_all=self.cover_all)


class TraceableReLU(TraceableFunctionWrapper):
    def __init__(self, *args, **opt):
        super().__init__(*args, **opt)

        self._func = F.relu
        self.traceable_node_type = 'RELU'


class TraceableIdentity(TraceableNode):
    def __init__(self):
        super().__init__()
        self.traceable_node_type = 'ID'

    def __call__(self, x, *arg, **opt):
        return x
