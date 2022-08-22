# Neural Network

A naive implementation of neural network. The code structure is heavily inspired by [PyTorch](https://github.com/pytorch/pytorch) and [TensorFlow](https://github.com/tensorflow/tensorflow). However, this package is used for educational purposes and is not intended to be adopted in production.

## Installation

```bash
git clone https://github.com/reshalfahsi/neuralnetwork
cd neuralnetwork
pip install .
```

## Quick Demo

```python
import neuralnetwork.nn as nn
import numpy as np

input = np.random.randn(1, 200)
m = nn.Linear(200, 100)
out = m(input)

print(out.shape)
# (1, 100)
```
