from backward_layers import *
from forward_layers import *


def build_network(model, hyper_parameters):

    weights = {}
    weight_scale = hyper_parameters["weight_scale"]

    for index, layer in enumerate(model, start=1):

        if layer["type"] == "convolution":

            filter_size = layer["filter_size"]
            depth = layer["depth"]
            filter_num = layer["filter_number"]

            weights["W%d" % index] = np.random.randn(filter_num, depth, filter_size, filter_size) * weight_scale
            weights["b%d" % index] = np.zeros(filter_num)

        elif layer["type"] == "fully_connected":

            hidden_size = layer["hidden_size"]
            depth = layer["depth"]

            weights["W%d" % index] = np.random.randn(hidden_size, depth) * weight_scale
            weights["b%d" % index] = np.zeros(depth)

    return weights


def forward(X, y, weights, model, reg=0.1):

    cache = {}
    previous_layer_type = "convolution"

    for index, layer in enumerate(model, start=1):

        weight = weights["W%d" % index]
        bias = weights["b%d" % index]
        input = X
        if index != 1:
            input = cache["layer_%d" % (index-1)][-1]


        if layer["type"] == "convolution":

            convolution = convolution_forward(input, weight, bias, padding=layer["padding"])
            activation = relu_forward(convolution)
            pool = max_pool_forward(activation)

            cache["layer_%d" % index] = [convolution, activation, pool]

            previous_layer_type = layer["type"]

        elif layer["type"] == "fully_connected":

            if previous_layer_type == "convolution":
                input = input.flatten()

            z = np.dot(input, weight) + bias
            a = relu_forward(z)

            cache["layer_%d" % index] = [z, a]

            previous_layer_type = layer["type"]

    last_layer = cache["layer_%d" % len(model)]

    scores = softmax(last_layer[-1])

    cost = -np.log(scores[y])

    sum_loss = cost + (0.5 * np.sum([np.sum(np.square(weights["W%d" % i])) for i in xrange(1, len(model) + 1)]) * reg)
    
    return scores, cache, sum_loss

def backward(x, y, scores, cache, weights, hyper_parameters, model):

    reg = hyper_parameters["regularization"]

    scores[y] -= 1
    deltas = [scores]
    grads = {}

    for i in reversed(xrange(len(model))):

        layer = i + 1
        delta = deltas[-1]

        if model[i]["type"] == "convolution":

            if model[i+1]["type"] == "fully_connected":
                ww = cache["layer_%d" % layer][-1].shape[1]
                hh = cache["layer_%d" % layer][-1].shape[2]
                delta = delta.reshape(model[i]["filter_number"], hh, ww)

            delta_max = max_pool_backward(delta, cache["layer_%d" % layer][1])
            delta_sig = relu_backwards(delta_max, cache["layer_%d" % layer][0])

            if layer == 1:
                input = x
            else:
                input = cache["layer_%d" % (layer - 1)][-1]

            d, dw, db = convolve_backward(delta_sig, input, weights["W%d" % layer], weights["b%d" % layer], padding=model[i]["padding"])

            deltas.append(d)

            grads["W%d" % layer] = dw
            grads["b%d" % layer] = db


        elif model[i]["type"] == "fully_connected":

            new_delta = delta

            if model[i]["activation"] == "relu":
                sigmoid_d = relu_backwards(new_delta, (cache["layer_%d" % layer][0]))
                delta = np.multiply(delta, sigmoid_d)

            act = cache["layer_%d" % (layer-1)][-1]

            if model[i-1]["type"] == "convolution":
                shape = len(act.flatten())
                act = act.reshape(shape,1)

            grads["W%d" % layer] = np.dot(act.reshape(act.shape[0], 1), delta.reshape(delta.shape[0],1).T)
            grads["b%d" % layer] = delta

            new_delta = np.dot(weights["W%d" % layer], delta)


            deltas.append(new_delta)

    return grads


def validate(data, weights, model):

    dataa = np.array([(data[i][0].reshape((28, 28, 1)), data[i][1]) for i in range(len(data))])
    count = 0
    for x in dataa:
        p = forward(x[0].reshape(1,28,28), x[1], weights, model)[0]
        max = np.argmax(p)
        if(max == x[1]):
            count = count + 1

    result = count / float(len(data))

    return result



def train(model, training_data, epochs, weights, hyper_parameters, test_data):

    reg = hyper_parameters["regularization"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]

    # data in form of (x,y) list of tuples
    data = training_data
    for i in xrange(epochs):
        import random
        random.shuffle(data)
        batches = [data[k:k + batch_size] for k in xrange(0, len(data), batch_size)]

        epoch_losses = 0
        batch_number = 0

        for batch in batches:

            grads = {}
            batch_loss = 0

            for index, layer in enumerate(model, start=1):
                if layer["type"] == "convolution":
                    grads["W%d" % index] = np.zeros((layer["filter_number"], layer["depth"], layer["filter_size"], layer["filter_size"]))
                    grads["b%d" % index] = np.zeros(layer["filter_number"])
                elif layer["type"] == "fully_connected":
                    grads["W%d" % index] = np.zeros((layer["hidden_size"], layer["depth"]))
                    grads["b%d" % index] = np.zeros(layer["depth"])

            for x, y in batch:
                image = x.reshape((28, 28)).reshape(1,28,28)
                scores, cache, loss = forward(image, y, weights, model, reg=reg)

                gradients = backward(image, y, scores, cache, weights, hyper_parameters, model)

                for index, layer in enumerate(model, start=1):
                    grads["W%d" % index] += reg * gradients["W%d" % index]
                    grads["b%d" % index] += reg * gradients["b%d" % index]

                batch_loss += loss

            batch_loss = batch_loss / batch_size

            epoch_losses += batch_loss
            batch_number += 1

            for index, layer in enumerate(model, start=1):
                weights["W%d" % index] = weights["W%d" % index] - learning_rate * (grads["W%d" % index] / batch_size)
                weights["b%d" % index] = weights["b%d" % index] - learning_rate * (grads["b%d" % index] / batch_size)

        print 'loss: %f' % (epoch_losses / batch_number)

    print validate(test_data, weights, model)


    return weights
