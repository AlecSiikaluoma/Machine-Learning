from utils import *


def convolution_forward(X, weights, bias, padding=0, stride=1):
    fn, fd, fh, fw = weights.shape
    xd, xh, xw = X.shape

    padded = X

    if padding > 0:
        padded = np.zeros((xd, xh + padding * 2, xw + padding * 2))
        _, pH, pW = padded.shape
        padded[:, padding:pH - padding, padding:pW - padding] = X

    oh = ((xh - fh + 2 * padding) / stride) + 1
    ow = ((xw - fw + 2 * padding) / stride) + 1
    od = fn

    # IM2COL:
    X_col = np.zeros((fd * fh * fw, oh * ow))
    W_row = np.zeros((fn, fd * fh * fw))

    for w in range(fn):
        W_row[w, :] = weights[w, :, :, :].flatten()
    for i in xrange(oh):
        for j in xrange(ow):
            X_col[:, (i * oh) + j] = padded[:, i:fh + i, j:fw + j].flatten()

    dot = (np.dot(W_row, X_col) + bias.reshape(bias.shape[0], 1)).reshape((fn, oh, ow), order='F')

    return dot


def max_pool_forward(x):
    D, H, W = x.shape
    out = np.zeros((D, H / 2, W / 2))

    for d in xrange(D):
        for h in xrange(0, H, 2):
            for w in xrange(0, W, 2):
                field = x[d, h:h + 2, w:w + 2]
                max = np.max(field)
                out[d, h / 2, w / 2] = max

    return out

def relu_forward(x):
    output = np.maximum(0, x)
    return output
