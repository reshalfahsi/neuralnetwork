import numpy as np
import scipy.special
import matplotlib.pyplot

%matplotlib inline

class NeuralNetwork:
    def __init__(self,input_node, hidden_node, output_node):
        #initialize neural network architecture
        self.inode = input_node
        self.hnode = hidden_node
        self.onode = output_node

        self.wih = np.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who = np.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))

    def compile(self, learning_rate, activation_function, epoch, batch_size = 1):
        #set hyperparameter
        self.lr = learning_rate
        self.activation = lambda x: scipy.special.expit(x)
        #self.activation = activation_function
        self.batch = batch_size
        self.epoch = epoch

    def fit(self,input,target):
        # train the network
        input_array_raw = np.array(input)
        target_array_raw = np.array(target)
        data = []
        for dx, dy in input_array_raw, target_array_raw:
            data.append(zip(dx,dy))
        data = np.array(data)

        for e in xrange(self.epoch):
            array_batched = np.random.choice(data,int(len(data)/self.batch))

            for input_array, target_array in array_batched:
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

    def predict(self,input):
        #predict output from input
        input_array = np.array(input).T

        hidden_input = np.dot(self.wih,input_array)
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation(final_input)

        return final_output

def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.01
    epoch = 10

    nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes)
    nn.compile(learning_rate=learning_rate,epoch=epoch)


if __name__=='__main__':
    main()