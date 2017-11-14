import numpy


def variable2array(obj):
    if isinstance(obj, numpy.ndarray):
        return obj
    elif hasattr(obj, 'data'):
        return obj.data
    else:
        raise TypeError('object does not have the way to convert into array')
