"""
Iterators of the Region of Interests (ROI)
which abstract the sliding window in images.
"""

import copy


class BaseROIIterator:
    """
    A base class of all the ROI abstraction.
    This provide a *parameterized* ROI abstraction
    which can be steered with the positions and indexes.
    It can be interpreted as a naive ROI generator.

    Args:
        x_begin (int): initial position of x-axis (column)
        y_begin (int): initial position of y-axis (row)
        x_end (int): end position of x-axis
        y_end (int): end position of y-axis
        roi_size (int or 2-D array-like): shape of the ROI
            that expect (width, height)
        stride(int or 2-D array-like; optional):
            spatial step of the ROI iteration
    """

    def __init__(self, x_begin, y_begin, x_end, y_end, roi_size, stride=1, **opt):
        self.x_begin = x_begin
        self.y_begin = y_begin
        self.x_end = x_end
        self.y_end = y_end

        self.current_x = self.x_begin
        self.current_y = self.y_begin

        if isinstance(roi_size, int):
            self.shape = (roi_size, roi_size)
        elif len(roi_size) == 2:
            self.shape = tuple(roi_size)
        else:
            raise TypeError('parameter roi_size must be an integer or '
                            '2-D array-like object '
                            'but {} was given.'.format(type(roi_size)))

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif len(stride) == 2:
            self.stride = tuple(stride)
        else:
            raise TypeError('parameter stride must be an intger or '
                            '2-D array-like object '
                            'but {} was given.'.format(type(roi_size)))

    def __next__(self):
        if self.current_y > self.y_end:
            raise StopIteration()

        ret = self.roi

        self.current_x += self.stride[0]

        if self.current_x > self.x_end:
            self.current_x = self.x_begin
            self.current_y += self.stride[1]

        return ret

    next = __next__

    def __iter__(self):
        return self

    @property
    def roi(self):
        return self.current_x, self.current_y, \
            self.current_x + self.shape[0], self.current_y + self.shape[1]


class DeformableROIIterator(BaseROIIterator):
    """
    An ROI abstraction which has a variable ROIs in the loop.
    ROIs will varies at near the edge.

    Args:
        width (int): width (number of columns) of the image
        height (int): height (number of rows) of the image
        roi_size (int): length of the ROI side
        stride (int): spatial step of the ROI iteration.
    """

    def __init__(self, width, height, roi_size, stride=1):
        self.width = width
        self.height = height

        if roi_size % 2 != 1:
            raise ValueError('roi_size must be a odd integer to define a '
                             'center of the region')

        super().__init__(int((1 - roi_size) / 2), int((1 - roi_size) / 2),
                         int(width - 1 - (roi_size - 1) / 2),
                         int(height - 1 - (roi_size - 1) / 2),
                         roi_size, stride)

    @property
    def roi(self):
        x0, y0, x1, y1 = super().roi

        if x0 < 0:
            x0 = 0

        if y0 < 0:
            y0 = 0

        if x1 >= self.width:
            x1 = self.width

        if y1 >= self.height:
            y1 = self.height

        return x0, y0, x1, y1


class InnerROIIterator(BaseROIIterator):
    """
    An ROI abstraction which has an inner sliding window.
    This provide a fix-sized square ROI.

    Args:
        width (int): width (number of columns) of the image
        height (int): height (number of rows) of the image
        roi_size (int): length of the ROI side
        stride (int): spatial step of the ROI iteration.
    """

    def __init__(self, width, height, roi_size, stride=1):
        self.width = width
        self.height = height

        super().__init__(0, 0, width - roi_size, height - roi_size,
                         roi_size, stride)


class ROIGenerator:
    """
    Wrapper for the image to generate a ROI from an ROI iterator.
    """

    def __init__(self, img, roi_iterator, **opt):
        self.img = copy.deepcopy(img)
        if len(img.shape) == 2:
            # if grayscaled image were given
            height, width = img.shape
        elif len(img.shape) == 3:
            # colored
            height, width = img.shape[0:2]
        else:
            raise TypeError('Image has unrecognized shape')

        self.iter = roi_iterator(width=width, height=height, **opt)

    def __next__(self):
        x0, y0, x1, y1 = next(self.iter)

        return self.img[y0:y1, x0:x1]

    next = __next__

    def __iter__(self):
        return self
