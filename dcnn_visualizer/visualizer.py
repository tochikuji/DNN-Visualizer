"""
Base class of all activation visualizers.
"""

import numpy
from dcnn_visualizer.traceable_chain import TraceableChain


class ActivationVisualizer:
    """
    A base class of all visualizers.
    """

    def __init__(self, model: TraceableChain):
        """
        Args:
            model(TraceableChain): model to visualize
        """

        if not isinstance(model, TraceableChain):
            raise TypeError('the model must be an instance of TraceableChain.')

        self.model = model
        self.layers = model.layer_names

        self.img_ = None
        self.current_activations = None
        self._previous_activations = None
        self._previous_input = None

    def analyze(self, img, layer):
        # validate layer's name
        if layer not in self.layers:
            raise ValueError(f'specified layer "{layer}" is not pickable in the model.')

        # check whether the input is in a minibatch
        if len(img.shape) != 4:
            self.img_ = numpy.array([img])
        else:
            self.img_ = numpy.array(img)

        if self._previous_activations is None or img is not self._previous_input:
            self.current_activations = self.model(self.img_)
            self._previous_activations = self.current_activations
            self._previous_input = img
        else:
            self.current_activations = self._previous_activations
