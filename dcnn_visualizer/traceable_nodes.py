import chainer
import chainer.links as L
import chainer.functions as F


class TraceableNode:
    """
    Base class of traceable (invertible) nodes.
    """

    def __init__(self):
        self.traceable_node_type = None

    def __call__(self, *args, **opt):
        raise NotImplementedError()


class TraceableFunctionWrapper(TraceableNode):
    """
    Function wrapper for `chainer.functions` to make it a callable TraceableNode,
    which retains fixed arguments.
    """

    def __init__(self, *args, **opt):
        super().__init__()

        self._func = None

        self.args = args
        self.opt = opt

    def __call__(self, x):
        return self._func(x, *self.args, **self.opt)


class TraceableConvolution2D(L.Convolution2D, TraceableNode):
    """
    Traceable Convolution layer which perform as `chainer.links.Convolution2D`.
    """

    def __init__(self, *args, **opt):
        """
        Construct the convolution node.

        Args:
            *args: same as L.Convolution2D
            **opt: ditto
        """

        super().__init__(*args, **opt)
        self.traceable_node_type = 'CONV'


class TraceableLinear(L.Linear, TraceableNode):
    """
    Traceable affine (fully-connected) layer,
    which also performs as L.Linear.
    """

    def __init__(self, *args, **opt):
        """
        Construct the affine node.

        Args:
            *args: same as L.Linear
            **opt: ditto
        """

        super().__init__(*args, **opt)
        self.traceable_node_type = 'AFFINE'


class TraceableMaxPooling2D(TraceableFunctionWrapper):
    """
    Traceable max-pooling layer, which wraps F.max_pooling_2d as a callable object.
    """

    def __init__(self, ksize, stride=None, pad=0, cover_all=True, *args, **opt):
        """
        Args:
            ksize: kernel size of max pooling
            stride: step size of the pooling window
            pad: zero-padding size
            cover_all: whether the pooling window exactly covers entire regions
            *args: optional unnamed arguments
            **opt: optional named arguments
        """

        super().__init__(*args, **opt)

        self._func = F.max_pooling_2d
        self.traceable_node_type = 'MP'

        self.ksize = ksize
        if stride is not None:
            self.stride = stride
        else:
            self.stride = ksize
        self.pad = pad
        self.cover_all = cover_all

    def __call__(self, x, *args, **opt):
        """
        Call F.max_pooling_2d with given parameters

        Args:
            x: BCHW array to pool
            *args: optional unnamed arguments
            **opt: optional named arguments

        Returns:
            result of pooling

        """
        return self._func(x, ksize=self.ksize, stride=self.stride, pad=self.pad, cover_all=self.cover_all)


class TraceableReLU(TraceableFunctionWrapper):
    """
    Traceable rectified linear unit.
    """

    def __init__(self, *args, **opt):
        super().__init__(*args, **opt)

        self._func = F.relu
        self.traceable_node_type = 'RELU'


class TraceableIdentity(TraceableNode):
    """
    Identity map to make DCNNs a ring.
    """

    def __init__(self):
        super().__init__()
        self.traceable_node_type = 'ID'

    def __call__(self, x, *arg, **opt):
        return x
