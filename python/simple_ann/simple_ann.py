from numpy import exp, array, random, dot, argmax

#hyperparameter
input_node = 4
hidden_node = 10
output_node = 3
learning_rate = 0.01
epoch = 10000

#input
input = raw_input('Masukan Input: ')
input = [int(i) for i in input.split() if(i.isdigit())]
input = array(input,ndmin=2).T
print input

#output
output = raw_input('Masukan Output: ')
output = [int(o) for o in output.split() if(o.isdigit())]
output = array(output,ndmin=2).T
print output

#ann weight
input_to_hidden_weights = random.random((hidden_node,input_node)) - 0.5
hidden_to_output_weights = random.random((output_node,hidden_node)) - 0.5

#training
for ep in xrange(epoch):
    #forward
    output_hidden = 1 / (1 + exp(-(dot(input_to_hidden_weights,input))))
    output_ann = 1 / (1 + exp(-(dot(hidden_to_output_weights,output_hidden))))

    #backpropagation
    error_output = output - output_ann
    error_hidden = dot(hidden_to_output_weights.T,error_output)
    hidden_to_output_weights += learning_rate*dot(error_output * output_ann * (1 - output_ann),output_hidden.T)
    input_to_hidden_weights += learning_rate*dot(error_hidden * output_hidden * (1 - output_hidden),input.T)

#evaluation
print ("True Output: {}\nArtificial Neural Network Output: {}\nAccuracy: {}%".format(output.T,output_ann.T,(1-(output[argmax(output)]-output_ann[argmax(output)]))[0]*100.0))
