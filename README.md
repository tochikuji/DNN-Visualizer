# DNN-Visualizer

## Description

Implementations of DCNN activation visualizing methods with chainer DNN framework.

![vis](https://user-images.githubusercontent.com/851759/36014547-d6bc7520-0dad-11e8-93ad-c87adc058bb0.png)

*The visualization examples of the feature representation of AlexNet's Conv4 10-th neuron*  
*Top left: Input (so cute!), Top right: DeconvNet, Bottom left: SaliNet, Bottom right: DeSaliNet*

## Introduction
This library contains visualization method for neuronal activation of deep convolutional neural networks (DCNNs).  
This library consists of [chainer](https://github.com/chainer/chainer).

Following visualization methods are implemented as:)

- Occulusion Saliency
- BackwardNets [1]
  - SaliNet [2]
  - DeconvNet [3]
  - DeSaliNet [1]

## Requirements

- Python 3.6+
- chainer 2.0+
- chainercv

## Installation

This is currently not available in PyPI.  
Thus install manually:

```sh
git clone https://github.com/tochikuji/DNN-Visualizer
cd DNN-Visualizer
pip install .
```

## Usage
### dcnn_visualizer.traceable_chain.TraceableChain
Traceable and invertible `chainer.Chain` which retains all intermediate activations.  
This lets us write a sequential forward propagation (`chainer.Chain.__call__`) as a composition of each function chains, 
in the same manner as `chainercv.PickableSequentialChain`, e.g. :

```python
class TraceableMLP(TraceableChain):
    def __init__(self):

        super().__init__()

        with self.init_scope():
            self.fc1 = TraceableLinear(10, 10)
            self.fc1_relu = F.relu
            self.fc2 = TraceableLinear(10, 10)
            self.fc2_relu = F.relu
            self.fc3 = TraceableLinear(10, 1)
            self.fc3_sigm = F.sigmoid

model = TraceableMLP()
# list named layer
layers = model.layer_names
# calculate forward propagation and retain all the intermedate activations
v = model(x)

print({layers[i]: y for i, y in enumerate(v)})
```

### dcnn_visualizer.desalinet

Backward propagation based visualization methods, 
including Machendran's DeSaliNet[1], Zeiler's DeconvNet[3] and Simonyan's SaliNet[2].

The main idea of these methods is to reconstruct an input component which affects the feature representations by using inverse operations of each layer [1].

![desali](https://user-images.githubusercontent.com/851759/36014667-567b7b6c-0dae-11e8-9631-265e46e86fac.png)

These methods require that the model should be an instance of `TraceableChain` and all nodes, without ignored ones, should be inherited from  `dcnn_visualizer.traceable_nodes.TraceableNode`.

```python
import dcnn_visualizer.traceable_nodes as tn


class SimpleCNN(TraceableChain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.conv1 = tn.TraceableConvolution2D(3, 10, 3)
            self.conv1_relu = tn.TraceableReLU()
            self.conv1_mp = tn.TraceableMaxPooling2D(ksize=2)
            # This LRN layer will be ignored in backward calculation
            # because it is not TraceableNode
            self.conv1_lrn = F.local_response_normalization

            self.conv2 = tn.TraceableConvolution2D(10, 5, 3)
            self.conv2_relu = tn.TraceableReLU()
            self.conv2_mp = tn.TraceableMaxPooling2D(ksize=2)

            self.fc3 = tn.TraceableLinear(None, 32)
            self.fc3_relu = tn.TraceableReLU()

            self.fc4 = tn.TraceableLinear(None, 10)
            self.fc4_relu = tn.TraceableReLU()


model = SimpleCNN()

visualizer = DeSaliNet(model)
# visualizer = SaliNet(model)
# visualizer = DeconvNet(model)

# visualize the 10-th neuron of conv2 layer 
visualized = visualizer.analyze(img, 'conv2', 10)
```

### dcnn_visualizer.occlusion.SaliencyVisualizer
Saliency visualization w.r.t. local occlusions.

## License
Apache License 2.0

Refer to `LICENSE` for more details.

## References
[1] Mahendran, A. and Vedaldi, A.: Salient deconvolutional networks, European Conference on Computer Vision, Springer, pp. 120–135 (2016).  
[2] Simonyan, K., Vedaldi, A. and Zisserman, A.: Deep inside convolutional networks: Visualising image classification models and saliency maps, arXiv preprint arXiv:1312.6034 (2013).  
[3] Zeiler, M. D. and Fergus, R.: Visualizing and understanding convolutional networks, European Conference on Computer Vision, Springer, pp. 818–833 (2014).

## Author

Aiga SUZUKI <tochikuji@gmail.com>
