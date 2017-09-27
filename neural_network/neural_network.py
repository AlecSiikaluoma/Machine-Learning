from utils import *
import numpy as np

class NeuralNetwork(object):

    def __init__(self, data, sizes):

        self.sizes = sizes
        self.data = data
        self.num_examples = len(data[0])
        self.layer_number = len(sizes[1:])

        # model containing the weights and biases.
        self.model = {}

        # initialize the model
        for index, layer_size in enumerate(sizes[1:], start=1):
            self.model["W%d" % index] = np.random.randn(sizes[index-1], layer_size)
            self.model["b%d" % index] = np.zeros(layer_size)

    def feed_forward(self, X):
        activations = [X]
        pre_activations = []
        for layer in xrange(1, self.layer_number + 1):
            W, b = self.model["W%d" % layer], self.model["b%d" % layer]
            z = np.dot(activations[layer-1], W) + b
            pre_activations.append(z)
            activations.append(sigmoid(z))
        scores = softmax(pre_activations[-1])

        return activations, pre_activations, scores

    def back_propagation(self, activations, scores, y):

        delta_last = scores
        delta_last[range(scores.shape[0]), y] -= 1
        deltas = [delta_last]

        gradients = {}

        #init gradients
        for index, layer_size in enumerate(self.sizes[1:], start=1):
            gradients["W%d" % index] = np.zeros((self.sizes[index-1], layer_size))
            gradients["b%d" % index] = np.zeros(layer_size)

        for layer in reversed(xrange(1, self.layer_number+1)):
            delta = deltas[self.layer_number-layer]
            gradW = np.dot(np.transpose(activations[layer-1]), delta)
            gradb = np.sum(delta, axis=0)

            new_delta = np.dot(delta, self.model["W%d" % layer].T)
            deltas.append(new_delta)

            gradients["W%d" % layer] += gradW
            gradients["b%d" % layer] += gradb

        return gradients

    def train(self, learning_rate=0.001, reg=0.1, mini_batch_size=40, epochs=4000):

        x = self.data[0]
        y = self.data[1]
        data = [(x[i], y[i]) for i in range(self.num_examples)]

        for e in xrange(epochs):
            import random
            random.shuffle(data)
            x = np.array([i[0] for i in data])
            y = np.array([i[1] for i in data])

            for batch in xrange(0, self.num_examples, mini_batch_size):
                X = x[batch:mini_batch_size]
                y = y[batch:mini_batch_size]
                a, z, s = self.feed_forward(X)
                grads = self.back_propagation(a, s, y)
                for layer in xrange(1, self.layer_number+1):
                    w = "W%d" % layer
                    b = "b%d" % layer

                    self.model[w] += -learning_rate * grads[w]
                    self.model[b] += -learning_rate * grads[b]

            if e % 1000 == 0:
                print "loss: %f" % (self.loss())


    def predict(self,x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        z1 = x.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        probs = softmax(z2)
        return np.argmax(probs, axis=1)

    def loss(self, reg=0.1):

        _,_,scores = self.feed_forward(self.data[0])

        log_prob = -np.log(scores[range(self.num_examples), self.data[1]])
        loss = np.sum(log_prob)
        sum = np.sum([np.sum(np.square(self.model["W%d" % i])) for i in xrange(1, self.layer_number+1)])

        loss += reg * sum

        return loss / self.num_examples

    def test(self, test_data):
        te = [(test_data[0][i], test_data[1][i]) for i in range(len(test_data[0]))]
        test_results = [(np.argmax(self.feed_forward(x)[2]), y) for (x, y) in te]
        return float(sum(int(x == y) for (x, y) in test_results)) / float(len(test_data[0]))


if __name__ == "__main__":

    data, validation, test = unpack_mnist()

    model = NeuralNetwork(data, [784,15,10])
    model.train()
    print model.test(test)
