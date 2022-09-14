# -*- coding: utf-8 -*-

"""## **Hyperparameters**"""

NUM_EPOCHS = 32
BATCH_SIZE = 16
lr = 1e-2

"""## **Dataset Preparation**

This tutorial will use a toy dataset from [MedMNIST](https://medmnist.com/). We use PneumoniaMNIST, which contains 2D X-ray image-label pairs for distinguishing between Pneumonia-infected and healthy lungs. The pneumonia-infected lung is denoted by the label `1` whilst the healthy lung is labeled as `0`.
"""

from neuralnetwork.ds.medmnist import PneumoniaMNIST
from neuralnetwork import ds
import numpy as np


train_dataset = PneumoniaMNIST(split="train", download=True)
test_dataset = PneumoniaMNIST(split="test", download=True)

print("Train Dataset:", len(train_dataset))
print("Test Dataset", len(test_dataset))

train_dataset.montage(length=20)

"""## **Neural Network**

The artificial neural network is a bio-inspired machine learning method that models neuronal signal propagation by matrix multiplication. Here we have two kinds of neuronal signal propagation: forward propagation and backward propagation. In forward propagation, the neuron actively conveys information from the "receptor" (or input) to the "central nervous system" (or output). Backward propagation or backpropagation, in short, is utilized in the training or learning process. In the learning process, the neural network transmits error gradients from the "central nervous system" to the "receptor". For further knowledge about the learning process, read more: [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) and [Backpropagation for a Linear Layer
](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html).
"""

import neuralnetwork.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralNetwork, self).__init__(**kwargs)
        self.linear0 = nn.Linear(784, 200, **kwargs)
        self.linear1 = nn.Linear(200, 200, **kwargs)
        self.linear2 = nn.Linear(200, 1, **kwargs)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.out0 = self.linear0(x)
        self.out1 = self.sigmoid(self.out0)
        self.out2 = self.linear1(self.out1)
        self.out3 = self.sigmoid(self.out2)
        self.out4 = self.linear2(self.out3)
        self.out5 = self.sigmoid(self.out4)

        return self.out5

    def backward(self, lr, criterion, method=None):
        # Computational Graph
        #
        self.dx0 = criterion.grad()  # loss_grad(pred, y)
        #        |
        self.dx1 = self.sigmoid.grad(self.out4)  # sigmoid_grad(pred)
        #        |
        #        +
        #       / \
        #      |   |
        #  b_grad  *
        #         / \
        #        |   |
        self.dx2 = self.linear2.grad(self.dx1 * self.dx0)  #   A_grad   x_grad
        #          .
        self.dx3 = self.sigmoid.grad(self.out2)  #          .
        self.dx4 = self.linear1.grad(self.dx3 * self.dx2)  #          .

        self.dx5 = self.sigmoid.grad(self.out0)
        self.dx6 = self.linear0.grad(self.dx5 * self.dx4)

        # -----------------------------------------------------

        if method == "newton":
            self.d2x0 = criterion.grad("hessian")
            self.d2x1 = self.sigmoid.grad(self.out4, "hessian")

            gradient = {
                "error_first": self.dx0,
                "error_second": self.d2x0,
                "nonlinearity_first": self.dx1,
                "nonlinearity_second": self.d2x1,
            }

            self.d2x2 = self.linear2.grad(gradient, "hessian")
            self.d2x3 = self.sigmoid.grad(self.out2, "hessian")

            gradient = {
                "error_first": self.dx2,
                "error_second": self.d2x2,
                "nonlinearity_first": self.dx3,
                "nonlinearity_second": self.d2x3,
            }

            self.d2x4 = self.linear1.grad(gradient, "hessian")
            self.d2x5 = self.sigmoid.grad(self.out0, "hessian")

            gradient = {
                "error_first": self.dx4,
                "error_second": self.d2x4,
                "nonlinearity_first": self.dx5,
                "nonlinearity_second": self.d2x5,
            }

            self.d2x6 = self.linear0.grad(gradient, "hessian")

        # -----------------------------------------------------

        self.linear0.update(lr, method)
        self.linear1.update(lr, method)
        self.linear2.update(lr, method)


"""## **Training**"""

import time


def accuracy(model, X, Y):
    pred = model(X)
    pred = pred > 0.5
    acc = np.sum(pred == Y)
    acc = acc / Y.shape[0]
    return acc


seed = np.random.randint(2147483647)
print(seed)

"""### **Gradient Descent**"""

train_loader = ds.get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = ds.get_loader(dataset=test_dataset, batch_size=1)

criterion = nn.MSELoss()
np.random.seed(seed)
model = NeuralNetwork()


start = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    loss = list()
    acc = list()
    for idx, pack in enumerate(train_loader):
        x, y = pack
        bs = x.shape[0]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(bs, 1, L) / 255.0
        y = y.reshape(bs, 1, 1)
        pred = model(x)
        loss.append(criterion(pred, y))
        model.backward(lr, criterion)
        acc.append(accuracy(model, x, y))
        if idx % 20 == 0 or idx == len(train_dataset) - 1:
            print(
                "{}/{} - The training loss at {}th epoch : {}  Training Accuracy:{}".format(
                    idx + 1,
                    len(train_dataset) // BATCH_SIZE,
                    epoch + 1,
                    np.array(loss).mean(),
                    np.array(acc).mean(),
                ),
            )
        if idx >= int(len(train_dataset) / BATCH_SIZE) - 1:
            break

    if np.array(acc).mean() > 0.9:
        break
    print("-----------------------------------------------------------")
end = time.perf_counter()

print(f"Training finished in {epoch + 1} epochs and {end - start:0.4f} seconds")

"""### **Newton Method**"""

train_loader = ds.get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = ds.get_loader(dataset=test_dataset, batch_size=1)

criterion = nn.MSELoss()
np.random.seed(seed)
model = NeuralNetwork()


start = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    loss = list()
    acc = list()
    for idx, pack in enumerate(train_loader):
        x, y = pack
        bs = x.shape[0]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(bs, 1, L) / 255.0
        y = y.reshape(bs, 1, 1)
        pred = model(x)
        loss.append(criterion(pred, y))
        model.backward(lr, criterion, "newton")
        acc.append(accuracy(model, x, y))
        if idx % 20 == 0 or idx == len(train_dataset) - 1:
            print(
                "{}/{} - The training loss at {}th epoch : {}  Training Accuracy:{}".format(
                    idx + 1,
                    len(train_dataset) // BATCH_SIZE,
                    epoch + 1,
                    np.array(loss).mean(),
                    np.array(acc).mean(),
                ),
            )
        if idx >= int(len(train_dataset) / BATCH_SIZE) - 1:
            break

    if np.array(acc).mean() > 0.9:
        break
    print("-----------------------------------------------------------")
end = time.perf_counter()

print(f"Training finished in {epoch + 1} epochs and {end - start:0.4f} seconds")

"""## **Testing**"""

import random

index = random.randint(0, len(test_dataset))

x, y = test_dataset[index]
x.resize((140, 140)).show()
x = np.array(x)
L = x.shape[0] * x.shape[1]
x = x.reshape(1, L) / 255.0
pred = model(x)

pred = pred.squeeze(0)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
print("Prediction: Pneumonia" if pred[0] else "Prediction: Healthy")
print("Ground Truth: Pneumonia" if y[0] else "Ground Truth: Healthy")
