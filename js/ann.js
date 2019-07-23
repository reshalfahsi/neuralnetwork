// Source : https://franpapers.com/en/machine-learning-ai-en/2017-neural-network-implementation-in-javascript-by-an-example/

function Neuron(id, layer, biais) {
    this.id = id;
    this.layer = layer;
    this.biais = biais || 0;
    this.dropped = false;
    this.output = undefined;
    this.error = undefined;
    this.activation = undefined;
    this.derivative = undefined;
};
////////////////////////////////////////////
function Network(params) {
    // Required variables: lr, layers
    this.lr = undefined; // Learning rate
    this.layers = undefined;
    this.hiddenLayerFunction = undefined; // activation function for hidden layer
    this.neurons    = undefined;
    this.weights    = undefined; 
    
    
    // ... load params
}
////////////////////////////////////////////
function randomBiais() {
    return Math.random() * 2 - 1;
}
function randomWeight() {
    return Math.random() * 2 - 1;
}
////////////////////////////////////////////
// Input layer filling
for (index = 0; index < this.layers[0]; index++)
    this.neurons[index].output = inputs[index];
// Fetching neurons from second layer (even if curr_layer equals 0, it'll be changed directly)
for (index = this.layers[0]; index < this.nbNeurons; index++)
{
    neuron = this.neurons[index];
    if (neuron.dropped)
        continue;
    // Update if necessary all previous layer neurons. It's a cache
    if (prev_neurons === undefined || neuron.layer !== curr_layer)
        prev_neurons = this.getNeuronsInLayer(curr_layer++);
    // Computing w1*x1 + ... + wn*xn
    for (sum = 0, n = 0, l = prev_neurons.length; n < l; n++) {
        if (!prev_neurons[n].dropped)
            sum += this.getWeight(prev_neurons[n], neuron) * prev_neurons[n].output;
    }
    // Updating output    
    neuron.output = neuron.activation(sum + neuron.biais); 
}
// Output layer error computing: err = (expected-obtained)
for (n = 0, l = outputs_neurons.length; n < l; n++)
{
    neuron = outputs_neurons[n];
    grad = neuron.derivative(neuron.output);
    err = targets[n] - neuron.output;
    neuron.error = grad * err;
    output_error += Math.abs(neuron.error);
    // Update biais 
    neuron.biais = neuron.biais + this.lr * neuron.error;        
}
this.outputError = output_error;
// Fetching neurons from last layer
for (index = this.layersSum[curr_layer-1] - 1; index >= 0; index--)
{
    neuron = this.neurons[index];
    // Dropping neuron is a technique to add dynamic into training
    if (neuron.dropped)
        continue;
    // Update if necessary all next layer neurons. It's a cache
    if (next_neurons === undefined || neuron.layer !== curr_layer)
        next_neurons = this.getNeuronsInLayer(curr_layer--);
    // Computing w1*e1 + ... + wn*en
    for (sum = 0, n = 0, l = next_neurons.length; n < l; n++) {
        if (!next_neurons[n].dropped)
            sum += this.getWeight(neuron, next_neurons[n]) * next_neurons[n].error;
    }
    // Updating error    
    neuron.error = sum * neuron.derivative(neuron.output); 
    this.globalError += Math.abs(neuron.error); 
    // Update biais
    neuron.biais = neuron.biais + this.lr * neuron.error;
    // Updating weights w = w + lr * en * output
    for (n = 0, l = next_neurons.length; n < l; n++)
    {
        if (next_neurons[n].dropped)
            continue;
        weight_index = this.getWeightIndex(neuron, next_neurons[n]); 
        // Update current weight
        weight = this.weightsTm1[weight_index] + this.lr * next_neurons[n].error * neuron.output;
        // Update maxWeight (for visualisation)
        max_weight = max_weight < Math.abs(weight) ? Math.abs(weight) : max_weight;
        // Finally update weights
        this.weights[weight_index] = weight;
    }
}

////////////////////// Main thread:
// Start web worker with training data through epochs
worker.postMessage({
    params: this.exportParams(),
    weights: this.exportWeights(),
    biais: this.exportBiais(),
    training_data: training_data,
    epochs: epochs
});
////////////////////// Worker:
// Create copy of our current Network
var brain = new Network(e.data.params);
brain.weights = e.data.weights;
// ...
// Feedforward NN
for (curr_epoch = 0; curr_epoch < epochs; curr_epoch++)
{
    for (sum = 0, i = 0; i < training_size; i++)
    {
        brain.feed(training_data[i].inputs);
        brain.backpropagate(training_data[i].targets);
        sum += brain.outputError;
    }
    
    global_sum += sum;
    mean = sum / training_size; 
    global_mean = global_sum / ((curr_epoch+1) * training_size); 
    // Send updates back to real thread
    self.postMessage({
        type: WORKER_TRAINING_PENDING,
        curr_epoch: curr_epoch,
        global_mean: global_mean,
    });
}
/////////////////////// Main thread:
// Training is over: we update our weights and biais
if (e.data.type === WORKER_TRAINING_OVER)
{
    that.importWeights( e.data.weights );
    that.importBiais( e.data.biais );
    // Feeding and bping in order to have updated values (as error) into neurons or others
    that.feed( training_data[0].inputs );
    that.backpropagate( training_data[0].targets );
}