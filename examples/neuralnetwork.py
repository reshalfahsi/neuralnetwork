# -*- coding: utf-8 -*-

"""## **Hyperparameters**"""

NUM_EPOCHS = 100
BATCH_SIZE = 16
lr = 1e-4

"""## **Dataset Preparation**"""

from neuralnetwork.ds.medmnist import PneumoniaMNIST
from neuralnetwork import ds
import numpy as np


train_dataset = PneumoniaMNIST(split='train', download=True)
test_dataset = PneumoniaMNIST(split='test', download=True)

print("Train Dataset:", len(train_dataset))
print("Test Dataset", len(test_dataset))

train_dataset.montage(length=20)

"""## **Neural Network**"""

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
    
    def backward(self, lr, criterion):
                                                               # Computational Graph
                                                               #
        self.dx0 = criterion.grad()                            # loss_grad(pred, y)
                                                               #        |
        self.dx1 = self.sigmoid.grad(self.out4)                # sigmoid_grad(pred)
                                                               #        |
                                                               #        +
                                                               #       / \
                                                               #      |   |
                                                               #  b_grad  *
                                                               #         / \
                                                               #        |   |
        self.dx2 = self.linear2.grad(grad=self.dx1* self.dx0)  #   A_grad   x_grad
                                                               #          .
        self.dx3 = self.sigmoid.grad(self.out2)                #          .
        self.dx4 = self.linear1.grad(grad=self.dx3 * self.dx2) #          .

        self.dx5 = self.sigmoid.grad(self.out0)
        self.dx6 = self.linear0.grad(grad=self.dx5 * self.dx4)

        self.linear0.update(lr)
        self.linear1.update(lr)
        self.linear2.update(lr)

"""## **Training**"""

def accuracy(model, X, Y):
    pred = model(X)
    pred = pred > 0.5
    acc = np.sum(pred == Y)
    acc = acc / Y.shape[0]
    return acc

train_loader = ds.get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = ds.get_loader(dataset=test_dataset, batch_size=1)

criterion = nn.MSELoss()
model = NeuralNetwork()

for epoch in range(NUM_EPOCHS):
    loss = list()
    acc = list()
    for idx, pack in enumerate(train_loader):
        x, y = pack
        bs = x.shape[0]
        L = x.shape[1] * x.shape[2]
        x = x.reshape(bs, L) / 255.0
        pred = model(x)
        loss.append(criterion(pred, y))
        model.backward(lr, criterion)
        acc.append(accuracy(model, x, y))
        print(
            "{}/{} - The training loss at {}th epoch : {}  Training Accuracy:{}".format(
                idx * BATCH_SIZE,
                len(train_dataset),
                epoch,
                np.array(loss).mean(),
                np.array(acc).mean(),
            ),
        )
        if idx > int(len(train_dataset) / BATCH_SIZE):
            break

    if np.array(acc).mean() > 0.9:
        break
    print("-----------------------------------------------------------")

print("Training finished in {} epochs".format(epoch))

"""## **Testing**"""

import random
from IPython.display import display

index = random.randint(0, len(test_dataset))

x, y = test_dataset[index]
display(x.resize((140, 140)))
x = np.array(x)
L = x.shape[0] * x.shape[1]
x = x.reshape(1, L)/255.
pred = model(x)

pred = pred.squeeze(0)
pred[pred>=0.5] = 1
pred[pred<0.5] = 0
print("Prediction: Pneumonia" if pred[0] else "Prediction: Healthy")
print("Ground Truth: Pneumonia" if y[0] else "Ground Truth: Healthy")
