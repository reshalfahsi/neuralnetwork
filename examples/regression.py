# -*- coding: utf-8 -*-
"""Regression.ipynb
"""

"""## **Hyperparameters**"""

NUM_EPOCHS = 32
BATCH_SIZE = 16
lr = 1e-2

"""## **Dataset Preparation**

This tutorial will use a regression problem dataset from [Medical Insurance Cost](https://www.kaggle.com/datasets/gauravduttakiit/medical-insurance-cost?select=Train_Data.csv). This dataset was inspired by the book *Machine Learning with R by Brett Lantz*. The data contains medical information and costs billed by health insurance companies. It contains 3630 rows of data and the following columns: `age`, `gender`, `BMI`, `children`, `smoker`, `region` and `charges`.
"""

from neuralnetwork.ds.medical_insurance_cost import MedicalInsuranceCost
from neuralnetwork import ds
import numpy as np


train_dataset = MedicalInsuranceCost(split='train')
test_dataset = MedicalInsuranceCost(split='test')

print("Train Dataset:", len(train_dataset))
print("Test Dataset", len(test_dataset))

train_dataset.head(n=10)

"""## **Neural Network**

The artificial neural network is a bio-inspired machine learning method that models neuronal signal propagation by matrix multiplication. Here we have two kinds of neuronal signal propagation: forward propagation and backward propagation. In forward propagation, the neuron actively conveys information from the "receptor" (or input) to the "central nervous system" (or output). Backward propagation or backpropagation, in short, is utilized in the training or learning process. In the learning process, the neural network transmits error gradients from the "central nervous system" to the "receptor". For further knowledge about the learning process, read more: [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) and [Backpropagation for a Linear Layer
](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html).
"""

import neuralnetwork.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralNetwork, self).__init__(**kwargs)
        self.linear0 = nn.Linear(3, 300, **kwargs)
        self.linear1 = nn.Linear(300, 30, **kwargs)
        self.linear2 = nn.Linear(30, 1, **kwargs)

        self.activation = nn.Tanh()

    def forward(self, x):
        self.out0 = self.linear0(x)
        self.out1 = self.activation(self.out0)
        self.out2 = self.linear1(self.out1)
        self.out3 = self.activation(self.out2)
        self.out4 = self.linear2(self.out3)
        self.out5 = self.activation(self.out4)

        return self.out5
    
    def backward(self, lr, criterion, method=None):
                                                               
        self.dx0 = criterion.grad()  
        self.dx1 = self.activation.grad(self.out4)
        self.dx2 = self.linear2.grad(self.dx1 * self.dx0)  
                                                             
        self.dx3 = self.activation.grad(self.out2)       
        self.dx4 = self.linear1.grad(self.dx3 * self.dx2)

        self.dx5 = self.activation.grad(self.out0)
        self.dx6 = self.linear0.grad(self.dx5 * self.dx4)

        if method == 'newton':
            self.d2x0 = criterion.grad('hessian')                                                        
            self.d2x1 = self.activation.grad(self.out4, 'hessian')

            gradient = {
                'error_first': self.dx0,
                'error_second': self.d2x0,
                'nonlinearity_first': self.dx1,
                'nonlinearity_second': self.d2x1,
            }               

            self.d2x2 = self.linear2.grad(gradient, 'hessian')
            self.d2x3 = self.activation.grad(self.out2, 'hessian') 

            gradient = {
                'error_first': self.dx2,
                'error_second': self.d2x2,
                'nonlinearity_first': self.dx3,
                'nonlinearity_second': self.d2x3,
            }         
                                                                            
            self.d2x4 = self.linear1.grad(gradient, 'hessian')
            self.d2x5 = self.activation.grad(self.out0, 'hessian')

            gradient = {
                'error_first': self.dx4,
                'error_second': self.d2x4,
                'nonlinearity_first': self.dx5,
                'nonlinearity_second': self.d2x5,
            } 

            self.d2x6 = self.linear0.grad(gradient, 'hessian')

        self.linear0.update(lr, method)
        self.linear1.update(lr, method)
        self.linear2.update(lr, method)

"""## **Utilities**"""

import time
import matplotlib.pyplot as plt

seed = np.random.randint(2147483647)
print(seed)

"""## **Gradient Descent**

### **Training**
"""

train_loader = ds.get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = ds.get_loader(dataset=test_dataset, batch_size=1)

criterion = nn.MSELoss()
np.random.seed(seed)
model = NeuralNetwork()

loss_train = list()

start = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    loss = list()
    for idx, pack in enumerate(train_loader):
        x, y = pack
        bs = x.shape[0]
        L = x.shape[1]
        x = x.reshape(bs, 1, L)
        y = y.reshape(bs, 1, 1)
        pred = model(x)
        loss.append(criterion(pred, y))
        model.backward(lr, criterion)
        if idx % 20 == 0 or idx == len(train_dataset) - 1:
            print(
                "{}/{} - The training loss at {}th epoch : {}".format(
                    idx + 1,
                    len(train_dataset) // BATCH_SIZE,
                    epoch + 1,
                    np.array(loss).mean(),
                ),
            )
        if idx >= int(len(train_dataset) / BATCH_SIZE) - 1:
            break

    loss_train.append(np.array(loss).mean())

    if np.array(loss_train).mean() < 0.01:
        break
    print("-----------------------------------------------------------")
end = time.perf_counter()

print(f"Training finished in {epoch + 1} epochs and {end - start:0.4f} seconds")

plt.title("Gradient Descent")
plt.plot(loss_train, color = 'r')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()
plt.clf()

"""### **Testing**"""

import random


mse = list()

# Evaluation
for idx, pack in enumerate(test_loader):
     x, y = pack
     bs = x.shape[0]
     L = x.shape[1]
     x = x.reshape(bs, 1, L)
     y = y.reshape(bs, 1, 1)
     x = test_dataset.final_pred(model(x))
     y = test_dataset.final_pred(y)
     mse.append(criterion(x, y))

     if idx >= int(len(test_dataset) / BATCH_SIZE) - 1:
         break


print(f"MSE on testing: {round(np.array(mse).mean() * 100., 2)}")


# Sample
index = random.randint(0, len(test_dataset))

x, y = test_dataset[index]
x = np.array(x)
L = x.shape[0]
x = x.reshape(1, 1, L)
pred = test_dataset.final_pred(model(x))
y = test_dataset.final_pred(y)

print(f"Prediction: ${round(pred[0,0,0], 2)}")
print(f"Ground Truth: ${round(y, 2)}")

"""## **Newton Method**

### **Training**
"""

train_loader = ds.get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = ds.get_loader(dataset=test_dataset, batch_size=1)

criterion = nn.MSELoss()
np.random.seed(seed)
model = NeuralNetwork()

loss_train = list()

start = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    loss = list()
    for idx, pack in enumerate(train_loader):
        x, y = pack
        bs = x.shape[0]
        L = x.shape[1]
        x = x.reshape(bs, 1, L)
        y = y.reshape(bs, 1, 1)
        pred = model(x)
        loss.append(criterion(pred, y))
        model.backward(lr, criterion, 'newton')
        if idx % 20 == 0 or idx == len(train_dataset) - 1:
            print(
                "{}/{} - The training loss at {}th epoch : {}".format(
                    idx + 1,
                    len(train_dataset) // BATCH_SIZE,
                    epoch + 1,
                    np.array(loss).mean(),
                ),
            )
        if idx >= int(len(train_dataset) / BATCH_SIZE) - 1:
            break

    loss_train.append(np.array(loss).mean())

    if np.array(loss_train).mean() < 0.01:
        break
    print("-----------------------------------------------------------")
end = time.perf_counter()

print(f"Training finished in {epoch + 1} epochs and {end - start:0.4f} seconds")

plt.title("Newton Method")
plt.plot(loss_train, color = 'r')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()
plt.clf()

"""### **Testing**"""

import random


mse = list()

# Evaluation
for idx, pack in enumerate(test_loader):
     x, y = pack
     bs = x.shape[0]
     L = x.shape[1]
     x = x.reshape(bs, 1, L)
     y = y.reshape(bs, 1, 1)
     x = test_dataset.final_pred(model(x))
     y = test_dataset.final_pred(y)
     mse.append(criterion(x, y))

     if idx >= int(len(test_dataset) / BATCH_SIZE) - 1:
         break


print(f"MSE on testing: {round(np.array(mse).mean() * 100., 2)}")


# Sample
index = random.randint(0, len(test_dataset))

x, y = test_dataset[index]
x = np.array(x)
L = x.shape[0]
x = x.reshape(1, 1, L)
pred = test_dataset.final_pred(model(x))
y = test_dataset.final_pred(y)

print(f"Prediction: ${round(pred[0,0,0], 2)}")
print(f"Ground Truth: ${round(y, 2)}")
