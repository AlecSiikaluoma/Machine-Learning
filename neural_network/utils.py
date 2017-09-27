import numpy as np
import os, sys

def unpack_mnist():
    import cPickle, gzip
    # Load the dataset
    f = gzip.open(os.path.join(sys.path[0], '../mnist.pkl.gz'), 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    ds = sigmoid(x)* (1 - sigmoid(x))
    return ds

def softmax(x):
    if len(x.shape) > 1:
        a = np.max(x, axis=1)
        x -= a.reshape((x.shape[0], 1))
        x = np.exp(x)
        a = np.sum(x, axis=1)
        x /= a.reshape((x.shape[0], 1))
    else:
        a = np.max(x)
        x -= a
        x = np.exp(x)
        a = np.sum(x)
        x /= a
    return x
