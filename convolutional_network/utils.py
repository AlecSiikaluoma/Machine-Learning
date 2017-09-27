import numpy as np
import os, sys


def zero_pad(x, padding):
    # input=28x28x1 --> out 32x32
    xd, xh, xw = x.shape
    padded = np.zeros((xd, xh + padding * 2, xw + padding * 2))
    _, pH, pW = padded.shape
    padded[:, padding:pH - padding, padding:pW - padding] = x

    return padded

def im2col(input, weigth, pixel_num, out_H, stride, filter_num, filter_size):
    X_col = np.zeros((pixel_num, out_H * out_H))
    W_row = np.zeros((filter_num, pixel_num))
    for w in range(filter_num):
        W_row[w, :] = weigth[:, :, :, w].flatten()
        for i in xrange(0, out_H, stride):
            for j in xrange(0, out_H, stride):
                X_col[:, (i*out_H)+j] = input[i:filter_size+i, j:filter_size+j, :].flatten()
    return X_col, W_row

def unpack_mnist():
    import cPickle, gzip
    # Load the dataset
    f = gzip.open(os.path.join(sys.path[0], '../mnist.pkl.gz'), 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)

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
