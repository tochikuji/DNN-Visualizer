"""
Activation visualizer based on activation saliency with an local occlusion.
This method sees the fluctuation of an neuronal activation (feature map) magnitude
by occluding a certain small region in the input images, corresponding to the receptive field.
This makes an heatmap of a variation of the each neuronal activations came from each local occlusions,
which implies a neuronal attentions (localities).
"""

from warnings import warn

import numpy
import dcnn_visualizer.tools as tools
from dcnn_visualizer.roi import DeformableROIIterator
import matplotlib.cm
from PIL import Image


class SaliencyMap:
    """
    SaliencyMap abstraction.

    Args:
        saliency (numpy.ndarray): Activation saliency tensor which has a shape like (height, width, channels)
    """
    def __init__(self, saliency):
        self.saliency = numpy.asarray(saliency)

        self.height, self.width, self.ch = self.saliency.shape

    @property
    def data(self):
        return self.saliency

    def heatmap(self, shape=None, cmap='bwr', value_range=None, centering=False, absolute=False):
        """
        Generate saliency heatmap corresponding to the object location.
        If you choose a colormap which has a transparency at the center of values,

        Args:
            shape (array_like): shape of heatmap, defaulting to same size as the saliency tensor
            cmap (str): the name of colormap definition in matplotlib.cm
            value_range (scalar or 2d_array_like): value range for normalize the saliency
            centering (bool): If True, heatmap will be symmetrized with 0
            absolute (bool): if True, saliency will be visualized with its magnitude

        Returns:
            heatmap (numpy.array)
        """

        saliency = self.saliency.copy()
        heatmap = numpy.zeros((*saliency.shape[:2], 3), dtype=numpy.uint8)

        if absolute:
            saliency = numpy.absolute(saliency)

        # validate arguments
        cm = getattr(matplotlib.cm, cmap, None)
        if cm is None:
            raise ValueError('invalid colormap name {}.'.format(cmap))

        # set shape as same spacial size as the saliency and take RGB-channel
        if shape is None:
            shape = (self.height, self.width, 3)
        elif len(shape) == 2:
            shape = (*shape, 3)
        else:
            raise TypeError("shape has an invalid shape. shape must be None or (h, w)")

        if value_range is None:
            sup = saliency.max()
            inf = saliency.min()
        elif len(value_range) == 2:
            inf, sup = value_range
        else:
            raise TypeError('value_range has invalid length that is not 2')

        # linear lookup
        if centering:
            absmax = max(abs(inf), abs(sup))
            saliency /= absmax
            saliency *= 127
            saliency += 128
        else:
            saliency -= saliency.min()
            saliency /= saliency.max()
            saliency *= 255

        # TODO: seek a faster way to concrete the heatmap
        for y, row in saliency:
            for x, value in row:
                color_rgba = cm(value, 1, True)
                heatmap[y, x] = color_rgba[:3]

        # resize heatmap to fit to specified shape
        if heatmap.shape != shape:
            pilimg = Image.fromarray(heatmap)
            pilimg.resize(shape[:2], Image.BICUBIC)
            heatmap = numpy.asarray(pilimg)

        return heatmap

    def compose(self, img, alpha=0.5, cmap='bwr', value_range=None, centering=False, absolute=False):
        """
        Blend a heatmap into the image.

        Args:
            img (numpy.ndarray): image to visualize the saliency which has a shape like (h, w, c)
            alpha (float): alpha value for alpha blending of the image and its saliency heatmap
            cmap (str): the name of colormap definition in matplotlib.cm
            value_range (scalar or 2d_array_like): value range for normalize the saliency
            centering (bool): If True, heatmap will be symmetrized with 0
            absolute (bool): if True, saliency will be visualized with its magnitude

        Returns:
            composite image (numpy.ndarray)
        """

        img_ = numpy.asarray(img)
        shape = img_.shape[:2]

        heatmap = self.heatmap(shape, cmap, value_range, centering, absolute)

        heatmap_pil = Image.fromarray(heatmap)
        img_pil = Image.fromarray(img_)

        visualized_pil = Image.blend(img_pil, heatmap_pil, alpha)

        return numpy.asarray(visualized_pil)


class SaliencyVisualizer:
    """
    Visualizer of network occlusion saliency.

    Args:
        model (chainer.Link): network to visualize an activation
        forward_func (function(numpy.ndarray, str) -> (numpy.ndarray)): model has no `forward` attribute, this will be
            used as an alternative to model.forward
    """

    def __init__(self, model, forward_func=None):
        self.model = model

        if hasattr(model, 'forward') and callable(model.forward):
            self.forward = lambda img, layer, args: model.forward(img, layer, *args)
        else:
            self.forward = lambda img, layer, args: forward_func(img, layer, *args)

    # noinspection PyTypeChecker
    def analyze(self, img, layer, ksize, stride=1, *forward_arg, **forward_opt):
        """
        Make a Saliency map of specified filter.

        Args:
            img (numpy.ndarray): An image we want to see a saliency (c, w, h)
            layer (str): Layer name of target layer; This must be meaningful for forward function.
            ksize (int): A size of occluding kernel in input image
            stride (int): Step size of occlusion i.e. a resolution of saliency analysis

        Returns:
            SaliencyMap
        """

        # validation for image
        if forward_arg is None:
            forward_arg = []

        if len(img.shape) == 3:
            cmap = 'color'

            height = img.shape[1]
            width = img.shape[2]

        elif len(img.shape) == 2:
            cmap = 'gray'

            height = img.shape[0]
            width = img.shape[1]

        else:
            raise TypeError("image has an unrecognized shape")

        # validate a layer name
        if not hasattr(self.model, layer):
            if hasattr(self.model, 'predictor'):
                warn("Note: the model seems to have an attribute 'predictor'."
                     "If you make a classifier through chainer.links.classifier, "
                     "pass the model.predictor as a 'model' argument.")

            raise ValueError('The model does not have the layer named "{}".'.format(layer))

        im_mean = numpy.mean(img)

        attention_layer = getattr(self.model, layer)
        n_feature_map = attention_layer.W.data.shape[0]

        activation_origin = self.forward(img, layer, *forward_arg, **forward_opt)
        activation_origin = tools.variable2array(activation_origin)
        activation_origin = activation_origin[0]
        # calculate L2-norm over all channels
        activation_magnitude_origin = numpy.linalg.norm(activation_origin, axis=(1, 2))

        saliency_map = numpy.zeros(int((height - 1) / stride) + 1, int((width - 1) / stride) + 1, n_feature_map)

        roi_iter = DeformableROIIterator(width, height, ksize, stride)
        for i, (x_begin, y_begin, x_end, y_end) in enumerate(roi_iter):
            occluded_img = numpy.copy(img)

            if cmap == 'gray':
                # noinspection PyUnusedLocal
                occlusion = occluded_img[y_begin:y_end, x_begin:x_end]
            else:
                # noinspection PyUnusedLocal
                occlusion = occluded_img[:, y_begin:y_end, x_begin:x_end]

            # noinspection PyUnusedLocal
            occlusion = im_mean

            activation_occluded = self.forward(occluded_img, layer, *forward_arg, **forward_opt)
            activation_occluded = tools.variable2array(activation_occluded)
            # ravel a mini-batch
            # noinspection PyUnusedLocal
            activation_occuluded = activation_occluded[0]
            # calculate L2-norm over all channels
            # noinspection PyTypeChecker
            activation_magnitude_occluded = numpy.linalg.norm(activation_occluded, axis=(1, 2))

            # here it'll get a variation between the original neuronal activations and occluded one
            # this indicates the local saliency of each neurons
            local_saliency = activation_magnitude_occluded - activation_magnitude_origin

            y_pos = int(i / saliency_map.shape[1])
            x_pos = i % saliency_map.shape[1]

            saliency_map[y_pos, x_pos] = local_saliency

        # concrete a SaliencyMap object
        return SaliencyMap(saliency_map)
