import numpy as np
from utils import *

def convolve_backward(delta, input, weigths, bias, padding=0, stride=1):
    fn, fd, fh, fw = weigths.shape
    xd, xh, xw = input.shape

    out_h = 1 + (xh + 2 * padding - fh) / stride
    out_w = 1 + (xw + 2 * padding - fw) / stride

    d_w = np.zeros_like(weigths)
    d_b = np.zeros_like(bias)

    pad_x = zero_pad(input, padding)
    pad_d_x = zero_pad(pad_x, padding)

    for filter in xrange(fn):
        for i in xrange(out_h):
            for j in xrange(out_w):
                field = pad_x[:, i:i + fh, j:j + fw]
                d_w[filter, :, :, :] += field * delta[filter, i, j]
                d_b[filter] += delta[filter, i, j]

                pad_d_x[:, i:i + fh, j:j + fw] += weigths[filter, :, :, :] * delta[filter, i, j]

    d_x = pad_d_x[:, padding:padding + xh, padding:padding + xw]

    return d_x, d_w, d_b


def max_pool_backward(delta, input):
    D, H, W = input.shape
    out = np.zeros((D, H, W))

    for d in xrange(D):
        for h in xrange(0, delta.shape[1], 2):
            for w in xrange(0, delta.shape[2], 2):
                field = input[d, h:h + 2, w:w + 2]
                max = np.max(field)
                delt = delta[d, h, w]
                cc = (field == max) * delt
                out[d, h:h + 2, w:w + 2] += cc

    return out


def relu_backwards(delta, input):
    x = input
    delt = np.array(delta, copy=True)
    delt[x <= 0] = 0
    return delt
