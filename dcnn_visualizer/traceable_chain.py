import contextlib

import chainer
from chainercv.links import PickableSequentialChain


class TraceableChain(PickableSequentialChain):
    """
    A traceable chain that can pick all intermediate layers and backtrace the calculation chains.

    The forward propagation must be sequential,
    that means forward propagation can be written as a composite of callable objects.

    >>> class MLP(TraceableChain):
    ...     def __init__(self):
    ...         with self.init_scope():
    ...             self.fc1 = L.Linear(None, 100)
    ...             self.fc1_relu = F.relu
    ...             self.fc2 = L.Linear(None, 10)
    ...             self.fc2_relu = F.relu
    ...             self.fc3 = L.Linear(None, 10)
    ...             self.fc3_pred = F.softmax

    The instance of TraceableChain is callable. Its `__call__` performs as a forward propagation.
    All intermediate activations will be retained automatically, and picked with `pick`, e.g.,

    >>> y = model(x)
    >>> act_fc2 = model.pick('fc2')
    """

    def __init__(self):
        super().__init__()

        self.pick = self.layer_names

    def renew_pick(self):
        self.pick = self.layer_names

    @contextlib.contextmanager
    def init_scope(self):
        try:
            with super().init_scope():
                yield
        finally:
            self.renew_pick()


if __name__ == '__main__':
    import chainer.functions as F
    import numpy
    from dcnn_visualizer.traceable_nodes import TraceableLinear

    class TraceableModel(TraceableChain):
        def __init__(self):

            super().__init__()

            with self.init_scope():
                self.fc1 = TraceableLinear(10, 10)
                self.fc1_relu = F.relu
                self.fc2 = TraceableLinear(10, 10)
                self.fc2_relu = F.relu
                self.fc3 = TraceableLinear(10, 1)
                self.fc3_sigm = F.sigmoid

    model = TraceableModel()

    x = numpy.random.rand(1, 10).astype('f')
    # noinspection PyTypeChecker
    v = model(x)

    print({model.layer_names[i]: y for i, y in enumerate(v)})
