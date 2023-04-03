# Neural Network

<div align="center">
  <a href="https://github.com/reshalfahsi/neuralnetwork/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"></a>
  <a href="https://github.com/reshalfahsi/neuralnetwork/actions/workflows/ci.yml"><img src="https://github.com/reshalfahsi/neuralnetwork/actions/workflows/ci.yml/badge.svg" alt="ci testing"></a>
</div>

A naive implementation of neural network. The code structure is heavily inspired by [PyTorch](https://github.com/pytorch/pytorch) and [TensorFlow](https://github.com/tensorflow/tensorflow). However, this package is used for educational purposes and is not intended to be adopted in production.

## Installation

```bash
git clone https://github.com/reshalfahsi/neuralnetwork
cd neuralnetwork
pip install .
```

## Quick Demo

Here is a short example of the usage of `neuralnetwork` components. For the complete demo, please take a look at [`examples/classification.py`](https://github.com/reshalfahsi/neuralnetwork/blob/main/examples/classification.py) and [`notebook/Classification.ipynb`](https://github.com/reshalfahsi/neuralnetwork/blob/main/notebook/Classification.ipynb) for the classification problem.

```python
import neuralnetwork.nn as nn
import numpy as np

input = np.random.randn(1, 1, 200)
m = nn.Linear(200, 100)
out = m(input)

print(out.shape)
# (1, 1, 100)
```
