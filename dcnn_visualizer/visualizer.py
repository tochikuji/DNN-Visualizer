"""
Base class of all activation visualizers.
"""


class ActivationVisualizer:
    """
    A base class of all visualizers.
    """

    def __init__(self, model, forward_func=None):
        """
        Args:
            model (chainer.Chain): model to visualize the activation
            forward_func (Optinal[Callable[(numpy.ndarray, str) -> numpy.ndarray]): if model has no `forward` attribute,
                this will be used as an alternative to model.forward
        """

        self.model = model

        if hasattr(model, 'forward') and callable(model.forward):
            self.forward = lambda img, layer, args: model.forward(img, layer, *args)
        else:
            self.forward = lambda img, layer, args: forward_func(img, layer, *args)
