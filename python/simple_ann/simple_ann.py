from numpy import exp, array, random, dot, genfromtxt
training_set_inputs, training_set_outputs = genfromtxt('input.txt', delimiter=' '),  array([genfromtxt('output.txt', delimiter=' ')]).T
synaptic_weights = 0.5 * random.random((len(training_set_inputs[0]),1)) - 1
for iteration in xrange(100000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print ("True Output {}\nArtificial Neural Network Output {}".format(training_set_outputs.T,output.T))
