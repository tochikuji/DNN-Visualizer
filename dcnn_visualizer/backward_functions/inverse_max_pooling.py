import numpy
import chainer.functions as F

from dcnn_visualizer.roi import InnerROIIterator, ROIGenerator, BaseROIIterator
from dcnn_visualizer.util import expect_ndarray


def sparse_max_unpooling(node, pooled, raw, is_positional):
    """

    Args:
        node (TraceableMaxUnpooling): MP node used in forward propagation
        pooled: Sub-sampled activation,
            here emphasize that this arg expects the data which has NCHW, i.e. in the mini-batch
        raw: the forward activation before pooling operation has been applied
        is_positional: if True, unpooling will be came in Zeiler's *strict* way.
            If not, the unpooling position will be center of receptive fields

    Returns:

    """

    return numpy.asarray([__sparse_max_unpooling_inner(node, pooled_item, raw_item, is_positional)
                          for pooled_item, raw_item in zip(pooled, raw)])


def __sparse_max_unpooling_inner(node, pooled_, raw_, is_positional):
    if isinstance(node.ksize, (tuple, list)) and len(set(node.ksize)) != 1:
        raise NotImplementedError('non-square pooling kernel is not supported.')

    if isinstance(node.ksize, (tuple, list)):
        ksize = node.ksize[0]
    else:
        ksize = int(node.ksize)

    with expect_ndarray(pooled_) as pooled, expect_ndarray(raw_) as raw:

        unpooled = numpy.zeros_like(raw, dtype=numpy.float32)
        roi_iter = InnerROIIterator(unpooled.shape[2], unpooled.shape[1], ksize, node.stride)

        for i, (x0, y0, x1, y1) in enumerate(roi_iter):
            receptive_field = unpooled[:, y0:y1, x0:x1]
            pooled_y, pooled_x = divmod(i, pooled.shape[2])

            if is_positional:
                receptive_field_raw = raw[:, y0:y1, x0:x1]
                max_locations = __max_location(receptive_field_raw)

                for ch_i, (py, px) in enumerate(max_locations):
                    unpooled[ch_i, y0 + py, x0 + px] = pooled[ch_i, pooled_y, pooled_x]

            else:
                # is not positional
                px, py = __center(x0, y0, x1, y1)
                unpooled[:, y0 + py, x0 + px] = pooled[:, pooled_y, pooled_x]

    return unpooled


def max_unpooling_locational(node, pooled, raw):
    return sparse_max_unpooling(node, pooled, raw, is_positional=True)


def max_unpooling_non_locational(node, pooled, raw):
    return sparse_max_unpooling(node, pooled, raw, is_positional=False)


def max_unpooling_diffusional(node, pooled, raw):
    return F.unpooling_2d(pooled, node.ksize, node.stride, node.pad, raw.shape)


def __center(x0, y0, x1, y1):
    return int((x0 + x1) / 2), int((y0 + y1) / 2)


def __max_location(receptive_field_raw_):
    with expect_ndarray(receptive_field_raw_) as receptive_field_raw:
        max_locations = [numpy.argwhere(c== c.max())[0] for c in receptive_field_raw]

    return tuple([loc.tolist() for loc in max_locations])
