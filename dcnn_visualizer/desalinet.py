"""
Implementation of Machendran's DeSaliNet[1] including Zeiler's deconvolutinal visualizing method (DeconvNet)[2],
and Simonyan's Network saliency (SaliNet)[3] as a special case.
This method is based on the back-propagation of the network activation similar to Zeiler's one.
DeSaliNet has a explicitness on its visualization result, but sometimes provide a propitious visualization to excess.

All of these methods requires that the network should be a "sequential",
that has no recursions, bypasses or something strange connections.
(e.g. LeNet, AlexNet, VGG. Not GoogLeNet, ResNet etc...)

[References]
[1] Mahendran, Aravindh, and Andrea Vedaldi. "Salient deconvolutional networks."
    European Conference on Computer Vision. Springer International Publishing, 2016.
[2] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks."
    European conference on computer vision. Springer, Cham, 2014.
    (https://arxiv.org/abs/1311.2901)
[3] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman.
    "Deep inside convolutional networks: Visualising image classification models and saliency maps."
    arXiv preprint arXiv:1312.6034 (2013).
    (https://arxiv.org/abs/1312.6034)
"""


from abc import ABCMeta

import numpy

from dcnn_visualizer.traceable_chain import TraceableChain
import dcnn_visualizer.traceable_nodes as tn
import dcnn_visualizer.backward_functions as bf


class BackwardNetBase(metaclass=ABCMeta):
    """
    Base class of backward-oriented activation visualizers (i.e. DeconvNet, SaliNet and DeSaliNet [1])
    """
    def __init__(self, model: TraceableChain):
        """

        Args:
            model:
        """

        if not isinstance(model, TraceableChain):
            raise TypeError('the model must be an instance of TraceableChain.')

        self.model = model
        self.layers = model.layer_names

        self._previous_activations = None
        self._previous_input = None

    def inverse_traceable_node(self, node, traced, raw):
        raise NotImplementedError()

    def analyze(self, img, layer, index=None, verbose=False):
        # validate layer's name
        if layer not in self.layers:
            raise ValueError(f'specified layer "{layer}" is not pickable in the model.')

        # check whether the input is in a minibatch
        if len(img.shape) != 4:
            img_ = numpy.array([img])
        else:
            img_ = numpy.array(img)

        if self._previous_activations is None or img is not self._previous_input:
            activations = self.model(img_)
            self._previous_activations = activations
            self._previous_input = img
        else:
            activations = self._previous_activations
            if verbose:
                print('forward propagation has been cached.')

        start_index = 0
        for layername in self.layers:
            if layer == layername:
                break
            else:
                start_index += 1

        if index is None:
            start_activation = activations[start_index]
        else:
            start_activation = numpy.zeros_like(activations[start_index]).astype('f')
            start_activation[:, index] = activations[start_index][:, index].data

        current_index = start_index
        current_activation = start_activation

        # backward propagation loop
        while True:
            current_attention_layer_name = self.layers[current_index]
            current_attention_layer = getattr(self.model, current_attention_layer_name)

            if current_index == 0:
                previous_activation = img_
            else:
                previous_activation = activations[current_index - 1]

            if isinstance(current_attention_layer, tn.TraceableNode):
                current_activation = self.inverse_traceable_node(current_attention_layer,
                                                                 current_activation,
                                                                 previous_activation)
            else:
                if verbose:
                    print(f'Named layer {current_attention_layer_name} has been ignored. '
                          'It is not an instance of TraceableNode.')

            current_index -= 1

            if current_index < 0:
                break

            # check shape of current_activation
            if current_activation.shape != activations[current_index].shape:
                raise ValueError('Shape of forward and backward were mismatched. '
                                 f'forward: {activations[current_index].shape}, '
                                 f'backward: {current_activation.shape}')

        return current_activation


class DeconvNet(BackwardNetBase):
    def __init__(self, model):
        super().__init__(model)

    def inverse_traceable_node(self, node, traced, raw):
        if isinstance(node, tn.TraceableConvolution2D):
            return bf.deconvolution2d(node, traced, raw)
        elif isinstance(node, tn.TraceableMaxPooling2D):
            return bf.max_unpooling_locational(node, traced, raw)
        elif isinstance(node, tn.TraceableReLU):
            return bf.inverse_relu_anew(node, traced, raw)
        elif isinstance(node, tn.TraceableLinear):
            return bf.inverse_linear(node, traced, raw)


class SaliNet(BackwardNetBase):
    def __init__(self, model):
        super().__init__(model)

    def inverse_traceable_node(self, node, traced, raw):
        if isinstance(node, tn.TraceableConvolution2D):
            return bf.deconvolution2d(node, traced, raw)
        elif isinstance(node, tn.TraceableMaxPooling2D):
            return bf.max_unpooling_non_locational(node, traced, raw)
        elif isinstance(node, tn.TraceableReLU):
            return bf.inverse_relu_locational(node, traced, raw)
        elif isinstance(node, tn.TraceableLinear):
            return bf.inverse_linear(node, traced, raw)


class DeSaliNet(BackwardNetBase):
    def __init__(self, model, locational_pooling=True):
        super().__init__(model)
        if locational_pooling:
            self.unpooling_fun = bf.max_unpooling_locational
        else:
            self.unpooling_fun = bf.max_unpooling_non_locational

    def inverse_traceable_node(self, node, traced, raw):
        if isinstance(node, tn.TraceableConvolution2D):
            return bf.deconvolution2d(node, traced, raw)
        elif isinstance(node, tn.TraceableMaxPooling2D):
            return self.unpooling_fun(node, traced, raw)
        elif isinstance(node, tn.TraceableReLU):
            return bf.relu(bf.inverse_relu_anew(node, traced, raw))
        elif isinstance(node, tn.TraceableLinear):
            return bf.inverse_linear(node, traced, raw)


if __name__ == '__main__':
    import chainer.functions as F
    import numpy as np

    class SimpleCNN(TraceableChain):
        def __init__(self):
            super().__init__()

            with self.init_scope():
                self.conv1 = tn.TraceableConvolution2D(3, 10, 3)
                self.conv1_relu = tn.TraceableReLU()
                self.conv1_mp = tn.TraceableMaxPooling2D(ksize=2)
                self.conv1_bn = F.local_response_normalization

                self.conv2 = tn.TraceableConvolution2D(10, 5, 3)
                self.conv2_relu = tn.TraceableReLU()
                self.conv2_mp = tn.TraceableMaxPooling2D(ksize=2)

                self.fc3 = tn.TraceableLinear(None, 32)
                self.fc3_relu = tn.TraceableReLU()

                self.fc4 = tn.TraceableLinear(None, 10)
                self.fc4_relu = tn.TraceableReLU()


    model = SimpleCNN()

    img = np.random.rand(1, 3, 28, 28).astype('f')

    visualizer = SaliNet(model)
    visualized_whole = visualizer.analyze(img, 'fc3', verbose=True)
    visualized_filter = visualizer.analyze(img, 'conv2', 1, verbose=True)
    print(visualized_whole.shape)
    print(visualized_filter.shape)
