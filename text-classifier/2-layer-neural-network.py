import numpy as np

# sigmoid function
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
x = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

# output dataset
y = np.array([ [0,0,1,1] ]).T

# seed random numbers to make calculation
# deterministic
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):

    # forward propagation
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))

    #loss gradient
    l1_error = y - l1
    l1_delta = l1_error * sigmoid(l1, True)

    syn0 += np.dot(l0.T, l1_delta)

print "output after training:"
print 'predict value:'
print l1
print 'true value:'
print y
