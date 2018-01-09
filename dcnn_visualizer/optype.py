from enum import Enum


class OpType(Enum):
    """
    Definitions of an element of the computation graph.

    ID : Identity map
    CONV : Convolution
    DECONV : Deconvolution
    RELU : Rectifier Linear Unit (ReLU)
    MP : Max-pooling
    AFFINE : Affine translation (fully-connected layer)
    """

    ID = 0
    CONV = 1
    DECONV = 2
    RELU = 3
    MP = 4
    AFFINE = 5

class BackwardType(Enum):
    """
    Definitions of the type of backward operations for each OpTypes.

    CONV_DEFAULT: Standard deconvolutional backwarding with the Hermite kernel
    MP_LOCATIONAL: Original Zeiler's Max-Unpooling backwarding which records max locations of
        the MP-receptive fields (max-location switches), and unpools into the region sparsely.
    MP_NON_LOCATIONAL: Max-Unpooling backwarding inspected in DeSaliNet paper. This unpooling way does not hold the
        max-locations and unpools into the specific region in contrast to the original Zeiler's way.
    MP_DIFFUSIONAL: [Experimental] Unpool the activation into the entire pooled region uniformly.
        It is mentioned that such unpooling way provides an obscure result.
    RELU_ANEW: Rectifies the rectified activation again, used in the original Zeiler's DeconvNet.
    RELU_LOCATIONAL: Backwards the rectified activation into the "positive location" in the previous activation.
        This implies to use the bottleneck information of the rectifying unit (DeSaliNet).
    AFFINE_TRANSPOSE: Straight way to backward the fully-connected layer.
    """

    CONV_DEFAULT = 10

    MP_LOCATIONAL = 20
    MP_NON_LOCATIONAL = 21
    MP_DIFFUSIONAL = 22

    RELU_ANEW = 30
    RELU_LOCATIONAL = 31

    AFFINE_TRANSPOSE = 40

    UNDEFINED = 99