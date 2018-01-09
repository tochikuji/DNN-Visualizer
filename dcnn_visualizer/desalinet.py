"""
Implementation of Machendran's DeSaliNet[1] including Zeiler's deconvolutinal visualizing method (DeconvNet)[2],
and Simonyan's Network saliency (SaliNet)[3] as a special case.
This method is based on the back-propagation of the network activation similar to Zeiler's one.
DeSaliNet has a explicitness on its visualization result, but sometimes provide a propitious visualization to excess.

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


from abc import ABCMeta, abstractmethod

from dcnn_visualizer.visualizer import ActivationVisualizer
from dcnn_visualizer.optype import OpType, BackwardType


class BackwardNetBase(ActivationVisualizer, metaclass=ABCMeta):
    """
    Base class of backward-oriented activation visualizers (i.e. DeconvNet, SaliNet and DeSaliNet [1])
    """
    def __init__(self, model, forward_func=None, opchain=None):
        """
        Args:
            model: model to visualize the activation
            forward_func: altenative to model.forward
            op_chain (OpChain):
        """

        super().__init__(model, forward_func)

        if opchain is None:
            raise ValueError('Operation chain is needed.')
        else:
            self.opchain = opchain

        self.opchain.set_forward(self.forward)

        # set an attribute
        # TODO: make it an abstract class
        self.backward_type = {}

    def analyze(self, img, layer, index=None, *forward_arg):
        # validate whether the layer name was linked with opchain layer
        if not self.opchain.haslayer(layer):
            raise NameError('operation chain has no layer {}'.format(layer))

        # result of the forward propagation
        # but each bottleneck information have not calculated yet,
        activation = self.forward(img, layer, *forward_arg)

        for layername, layertype in self.opchain.chain:
            if layertype == OpType.CONV:


class DeconvNet(BackwardNetBase):
    def __init__(self, model, forward_func=None, opchain=None):
        super().__init__(model, forward_func, opchain)

        self.opchain.backward_type = {
            OpType.CONV: BackwardType.CONV_DEFAULT,
            OpType.MP: BackwardType.MP_LOCATIONAL,
            OpType.RELU: BackwardType.RELU_ANEW,
            OpType.AFFINE: BackwardType.AFFINE_TRANSPOSE
        }

class SaliNet(BackwardNetBase):
    super().__init__(model, forward_func, opchain)

    self.opchain.backward_type = {
        OpType.CONV: BackwardType.CONV_DEFAULT,
        OpType.MP: BackwardType.MP_NON_LOCATIONAL,
        OpType.RELU: BackwardType.RELU_LOCATIONAL,
        OpType.AFFINE: BackwardType.AFFINE_TRANSPOSE
    }

class DeSaliNet(BackwardNetBase):
    super().__init__(model, forward_func, opchain)

    self.opchain.backward_type = {
        OpType.CONV: BackwardType.CONV_DEFAULT,
        OpType.MP: BackwardType.MP_LOCATIONAL,
        OpType.RELU: BackwardType.RELU_LOCATIONAL,
        OpType.AFFINE: BackwardType.AFFINE_TRANSPOSE
    }
