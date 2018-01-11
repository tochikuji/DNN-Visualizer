import chainer
import contextlib


@contextlib.contextmanager
def expect_ndarray(x):
    is_raveled = False

    try:
        if isinstance(x, chainer.Variable):
            x = x.data
            is_raveled = True

        yield x

    finally:
        if is_raveled:
            x = chainer.Variable(x)


if __name__ == '__main__':
    import numpy

    v = chainer.Variable(numpy.arange(10))

    with expect_ndarray(v) as data:
        data[0] = data.max()

