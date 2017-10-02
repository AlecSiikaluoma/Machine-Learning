from convolutional_net import *

######### Parameters #########
hyper_parameters = {}
hyper_parameters["weight_scale"] = 0.01
hyper_parameters["regularization"] = 0.1
hyper_parameters["learning_rate"] = 0.1
hyper_parameters["batch_size"] = 10
##############################


######### model ##############
model = []
model.append({
    "type":"convolution",
    "filter_size":5,
    "filter_number":20,
    "depth":1,
    "activation":"relu",
    "pooling":"max",
    "padding":2
})
model.append({
    "type":"convolution",
    "filter_size":5,
    "filter_number":16,
    "depth":20,
    "activation":"relu",
    "pooling":"max",
    "padding":2
})
hidden_size = 784 # This is the flattened size of the last convolutional layer before fully-connected net.
model.append({
    "type":"fully_connected",
    "hidden_size":hidden_size,
    "depth":30,
    "activation":"sigmoid"
})
model.append({
    "type":"fully_connected",
    "hidden_size":30,
    "depth":10,
    "activation":"scores"
})
###############################

########### DATA ##############
d, e, c = unpack_mnist()
x = d[0]
y = d[1]
tx = c[0]
ty = c[1]

test_data = np.array([(tx[i], ty[i]) for i in range(1000)])
# SELECTING ONLY 100 samples from the dataset for testing purposes. Change 100 to max 50000.
data = np.array([(x[i], y[i]) for i in range(100)])
###############################

# TESTS #
weights = build_network(model, hyper_parameters)
trained_weights = train(model, data, 5, weights, hyper_parameters, data)


#weights1 = build_network(model, hyper_parameters)
#trained_weights1 = train(model, data, 0, weights1, hyper_parameters)

# RESULTS: #
# Overfitting a tiny dataset results in: 0.65 / 1.0
# Overfitting a tiny dataset with random parameters results in: 0.09 / 1.0
