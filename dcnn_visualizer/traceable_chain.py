import contextlib

import chainer
from chainercv.links import PickableSequentialChain


class TraceableChain(PickableSequentialChain):
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
    from traceable_nodes import TraceableLinear

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
    v = model(x)

    print({model.layer_names[i]: y for i, y in enumerate(v)})
