import copy, numpy as np

np.random.seed(0)

# compute sigmoid nonlinerity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid result to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1-output)

# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
lr = 0.1 # learning rate
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(20000):

    # generate a simple addition problem (a+b=c)
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]

    # true ansewer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros((1, hidden_dim)))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):

        # generate input and output
        x = np.array([ [a[binary_dim-position-1], b[binary_dim-position-1]] ])
        y = np.array([ [c[binary_dim-position-1]] ]).T

        # hidden layer (input + prev_hidden)
        layer_1 = sigmoid(np.dot(x, synapse_0) +
                          np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # layer_2_deltas
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        # decode estimate so we can print it out
        d[binary_dim-position-1] = np.round(layer_2[0][0])

        
        # store hidden layer for gradient
        layer_1_values.append(copy.deepcopy(layer_1))

    # compute gradient for W
    for position in range(binary_dim):

        x = np.array([ [a[position], b[position]] ])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        # layer2(output) del_error
        layer_2_delta = layer_2_deltas[-position-1]
        # layer1(hidden) del_error
        layer_1_delta = layer_2_delta.dot(synapse_1.T)*sigmoid_output_to_derivative(layer_1)

        # update all weights
        synapse_1_update += layer_1.T.dot(layer_2_delta)
        synapse_h_update += prev_layer_1.T.dot(layer_1_delta)
        synapse_0_update += x.T.dot(layer_1_delta)

    synapse_0 += synapse_0_update * lr
    synapse_1 += synapse_1_update * lr
    synapse_h += synapse_h_update * lr

    # reset for next round to train
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    if j% 1000 == 0:
        print "Error:" + str(overallError)
        print 'pred:' + str(d)
        print 'True:' + str(c)

        out = 0
        for index, x in enumerate(reversed(d)):
            out += x*pow(2, index)
        print str(a_int) + ' + ' + str(b_int) + ' = ' + str(out)
        print '----------------'
