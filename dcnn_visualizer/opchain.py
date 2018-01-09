"""
Operation chain abstraction providing a relationship between layer name and type of link.
Used by activation back-propagation approaches.
"""

from dcnn_visualizer.optype import OpType, BackwardType


class OpChain:
    def __init__(self, model, name, type):
        self.model = model
        self.chain = list(zip(name, type))
        self._forward = None

        self.bottleneck_pool = dict()
        self.backward_type = {
            OpType.CONV: BackwardType.UNDEFINED,
            OpType.MP: BackwardType.UNDEFINED,
            OpType.RELU: BackwardType.UNDEFINED,
            OpType.AFFINE: BackwardType.UNDEFINED
        }

    def add_link(self, name, type):
        self.chain.append((name, type))

    def set_forward(self, forward_func):
        self._forward = lambda img, layer, args: forward_func(img, layer, *args)

    def forward(self, img, layer, volatile=True, *forward_arg):
        """
        Get the specific layer activation with forward propagation.

        Args:
            img (numpy.ndarray): image to get the activation
            layer (str): name of the layer to get the activation
            volatile (bool): If True, the progress of the forward propagation i.e. the bottleneck information
                to calculate the DeSali-backwarding will be renounced.

        Returns: specified layer activation (numpy.ndarray)
        """

        if volatile:
            return self._forward(img, layer, *forward_arg)

        for layername, layertype in self.chain:
            if layertype == OpType.CONV

    def backward(self, act):
        pass
