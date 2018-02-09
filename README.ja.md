## Description 
畳み込みニューラルネットワークの特徴表現を可視化するやつ。  
chainerで動く。

ここに画像がはいる


## Introduction
深層畳み込みニューラルネットワーク(DCNN)における種々の特徴表現の可視化手法を実装したライブラリです。  
DNNのフレームワークはchainerを想定しています。

現状実装されているのは

- Occulusion Saliency
- BackwardNets [1]
  - SaliNet [2]
  - DeconvNet [3]
  - DeSaliNet [1]

です。

## Requirements

- Python 3.6+
- chainer 2.0+
- chainercv

## Installation

まだPyPIにあげてないです。
たぶん上げる予定もないです。

```sh
git clone https://github.com/tochikuji/DNN-Visualizer
cd DNN-Visualizer
pip install .
```

## Contributions

### dcnn_visualizer.traceable_chain.TraceableChain
中間表現を保持、逆伝搬可能な`chainer.Chain`です。  
`chainercv.PickableSequentialChain`と同様にシーケンシャルな順伝搬処理を
合成関数の形でかけます。

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

活性マップの逆伝搬による勾配ベースの可視化手法です。  

ここにずがはいる

ベースはDeSaliNet[1]で、原論文と同様に $ReLU^{\dagger}$ の対応でZeiler et al. [3], Simonyan et al. [2]の
可視化手法も実装しています。  
これらの手法ではモデルは`TraceableChain`かつ、逆伝搬したいノードは`dcnn_visualizer.traceable_nodes.TraceableNode`でなければなりません。

```python
import dcnn_visualizer.traceable_nodes as tn


class SimpleCNN(TraceableChain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.conv1 = tn.TraceableConvolution2D(3, 10, 3)
            self.conv1_relu = tn.TraceableReLU()
            self.conv1_mp = tn.TraceableMaxPooling2D(ksize=2)
            # TraceableNodeでないものは逆伝搬で無視される
            self.conv1_bn = F.local_response_normalization

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

# conv2層の10番目のフィルタ活性を可視化
visualized = visualizer.analyze(img, 'conv2', 10)
```

### dcnn_visualizer.occlusion.SaliencyVisualizer
Local occulutionに対する活性マップの顕著性を見るものです。


## License
Apache License 2.0

詳細は`LICENSE`を参照のこと。

## References
[1] Mahendran, A. and Vedaldi, A.: Salient deconvolutional networks, European Conference on Computer Vision, Springer, pp. 120–135 (2016).  
[2] Simonyan, K., Vedaldi, A. and Zisserman, A.: Deep inside convolutional networks: Visualising image classification models and saliency maps, arXiv preprint arXiv:1312.6034 (2013).  
[3] Zeiler, M. D. and Fergus, R.: Visualizing and understanding convolutional networks, European Conference on Computer Vision, Springer, pp. 818–833 (2014).

## Author

Aiga SUZUKI <tochikuji@gmail.com>
