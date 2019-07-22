import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot

# Side Notes: Open Problem indicate we want make Neural Network architecture that look like tensorflow/keras instead of Tariq Rashid's

class NeuralNetwork:
    def __init__(self,input_node, hidden_node, output_node):
        #initialize neural network architecture
        self.inode = input_node
        self.hnode = hidden_node
        self.onode = output_node

        self.wih = np.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who = np.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))

    def compile(self, learning_rate, epoch, batch_size, activation_function = None):
        #set hyperparameter
        self.lr = learning_rate
        self.activation = lambda x: scipy.special.expit(x)
        #self.activation = activation_function
        self.batch = batch_size
        self.epoch = epoch

    def fit(self,input,target):
        input_array = np.array(input,ndmin=2).T
        target_array = np.array(target,ndmin=2).T

        #forward
        hidden_input = np.dot(self.wih,input_array)
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation(final_input)

        #backpropagation
        output_error = target_array - final_output
        hidden_error = np.dot(self.who.T,output_error)
        self.who += self.lr*np.dot((output_error*final_output*(1.0-final_output)),np.transpose(hidden_output))
        self.wih += self.lr*np.dot((hidden_error*hidden_output*(1.0-hidden_output)),np.transpose(input_array))


    def predict(self,input):
        #predict output from input
        input_array = np.array(input).T

        hidden_input = np.dot(self.wih,input_array)
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation(final_input)

        return final_output

    ''' Open Problem
    def fit(self,input,target):
        # train the network
        input_array_raw = np.asarray(input)
        target_array_raw = np.array(target)
        #print(input_array_raw)
        #print(type(input))
        #data = []
        #for dx, dy in input_array_raw, target_array_raw:
            #data.append(zip(dx,dy))
        #data = np.array(data)

        for e in xrange(self.epoch):
            #array_batched = np.random.choice(data,int(len(data)/self.batch))

            #for input_array, target_array in array_batched:
            for input_array, target_array in input_array_raw, target_array_raw:
                #forward
                hidden_input = np.dot(self.wih,input_array.T)
                hidden_output = self.activation(hidden_input)
                final_input = np.dot(self.who,hidden_output)
                final_output = self.activation(final_input)

                #backpropagation
                output_error = target_array.T - final_output
                hidden_error = np.dot(self,who.T,output_error)
                self.who += self.lr*np.dot((output_error*final_output*(1.0-final_output)),hidden_output.T)
                self.whi += self.lr*np.dot((hidden_error*hidden_output*(1.0-hidden_output)),input_array)
    '''


def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.01
    epoch = 10
    batch_size = 20

    ##############################################################################
    ''' Open Problem
    converters = {0: lambda s: float(s.strip('"'))}
    train_data = np.loadtxt("mnist_dummy.csv", delimiter=',', converters=converters)
    #train_data = np.array(open('mnist_dummy.csv').readline().split(',')[1:], int)
    print train_data

    Xtrain = []
    Ytrain = [] #np.array(open('mnist_dummy.csv').readline().split(',')[0:], int)
    print Ytrain
    for data in train_data:
        Ytrain.append(data[0:1:1])
        Xtrain.append(data[1:len(data):1])
        print data[1:len(data):1]
    
    print (np.array(Xtrain))
    
    test_data = np.empty(len(train_data),len(train_data[0]))
    with open("mnist_dummy.csv", "r") as file:
        for lines in file:
            dtd = [float(i) for line in lines for i in line.split(',') if i.strip()]
            test_data.append(dtd)
    
    print np.array(test_data)

    Ytest, Xtest = [float(y) for y in train_data[:][0]], [x for x in train_data[:][1:]]
    Ytest = np.eye(output_nodes)[Ytest]
    Xtest_copy = Xtest
    Xtest = []
    for x in Xtest:
        x = [float(dx)/255.0 for dx in x if(dx!=' ' and dx!='\n' and dx!=',')]
        Xtest.append(x)
    Xtest = np.array(Xtest)
    '''
    ################################################################################

    nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes)
    nn.compile(learning_rate=learning_rate,epoch=epoch,batch_size=batch_size)
    #nn.fit(Xtrain,Ytrain) # Open Problem

    training_data_file = open("mnist_test.csv",'r')
    training_data = training_data_file.readlines()
    training_data_file.close()

    for e in range(epoch):
        for data in training_data:
            all_data = data.split(',')
            input = (np.asfarray(all_data[1:])/255.0*0.99) + 0.01
            target = np.zeros(output_nodes) + 0.01
            target[int(all_data[0])] = 0.99
            nn.fit(input,target)

    #evaluate
    score = []

    ''' Open Problem
    for x,y in Xtest,Ytest:
        output = nn.predict(x)
        label = np.argmax(output)
        if label==y:
            score.append(1)
        else:
            score.append(0)
    '''

    test_data_file = open("mnist_dummy.csv",'r')
    test_data = test_data_file.readlines()
    test_data_file.close()

    for data in test_data:
        all_data = data.split(',')
        y = int(all_data[0])
        input = (np.asfarray(all_data[1:])/255.0*0.99) + 0.01
        output = nn.predict(input)
        label = np.argmax(output)
        #print ('predicted: {}, True: {}'.format(label,y))
        if label==y:
            score.append(1)
        else:
            score.append(0) 

    score = np.array(score)
    performance = float(score.sum())/float(score.size) * 100.0
    print ("Performance: {}%".format(performance))

if __name__=='__main__':
    main()
