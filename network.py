import numpy as np
import random
import math
import h5py


class model(object):

    def __init__(self, layers_dims, output_layer):
        """layers_dims is list containing the number of neurons in each layer, including the input layer, with
        length(number of hidden units + 1)
        Example:- if list was [2,3,3,1], input layer has 2 nodes, followed by 2 hidden layer with 3 nodes each
        and an output layer with 1 node
        output_layer :- takes value "sigmoid" if binary classification or "softmax" if multi-class classification
        """
        self.layers_dims = layers_dims
        self.parameters = {}
        self.output_layer = output_layer

    def initialize_parameters_deep(self, parameter_init='random'):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        parameter_init -- weight initialisation method - 'Xavier', 'random' - default -'random'

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(self.layers_dims)            # number of layers in the network

        if parameter_init == 'random':
            for l in range(1, L):
                # random initialisation scaled by 10
                self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1])*10
                self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
                assert(self.parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
                assert(self.parameters['b' + str(l)].shape == (self.layers_dims[l], 1))
        else:
            for l in range(1, L):
                # Xavier initialisation
                self.parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1])*np.sqrt(2/self.layers_dims[l-1])
                self.parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
                assert(self.parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
                assert(self.parameters['b' + str(l)].shape == (self.layers_dims[l], 1))

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[0,k*mini_batch_size:(k+1)*mini_batch_size].reshape(1, mini_batch_size)
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]
            mini_batch_Y = shuffled_Y[0,num_complete_minibatches*mini_batch_size:].reshape(1, m-(num_complete_minibatches*mini_batch_size))
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation or
        [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        output_layer -- softmax or sigmoid based on the classifier, default sigmoid

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(self.parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="relu")
            caches.append(cache)

        if self.output_layer == 'sigmoid':
            # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
            AL, cache = linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="sigmoid")
            caches.append(cache)
        else:
            # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
            AL, cache = linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation='softmax')
            caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y, lamda=0):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape-(1, number of examples) if binary classification
            or shape-(number of classes, number of examples) if multi-class classification
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples) if binary classification
            or one-hot encoded matrix of "labels" and shape (number of classes, number of examples) if multi-class classification,
            applicable when softmax=True
        softmax -- default False, set True if multi-class classification and pass Y as one-hot encoded matrix.

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        if self.output_layer == "sigmoid":
            # Compute loss from aL and y.
            cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        else:
            # Implement softmax cross entropy loss from aL and y.
            cost = (1./m) * (-np.dot(Y,np.log(AL).T))

        if lamda != 0:
            # add regularization cost if lamda != 0
            cost = cost + self.l2_cost(lamda=lamda, m=m)
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        return cost

    def l2_cost(self, lamda, m):
        L = len(self.layers_dims)
        reg_cost = 0
        for l in range(1, L):
            reg_cost += np.square(np.sum(self.parameters["W" + str(l)]))
        return reg_cost*(1/m)*(lamda/2)

    def L_model_backward(self, AL, Y, caches, lamda=0):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        if self.output_layer == 'sigmoid':
            # Initializing the backpropagation
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

            # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
            current_cache = caches[L-1]
            grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid", lamda=lamda)
        else:
            # initializing backprop for softmax
            current_cache = caches[L-1]
            l_cache, _ = current_cache
            A_prev, W, b = l_cache
            dZL = AL - Y
            grads["dW" + str(L)] = 1./m * np.dot(dZL,A_prev.T)
            if lamda!=0:
                # add regularization term in dW for last layer
                grads["dW" + str(L)] += (lamda/m)*W

            grads["db" + str(L)] = 1./m * np.sum(dZL, axis = 1, keepdims = True)
            grads["dA" + str(L-1)] = np.dot(W.T,dZL)


        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu", lamda=lamda)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters_with_gd(self, grads, learning_rate):
        """
        Implements parameter update in gradient descent
        """
        parameters = self.parameters
        L = len(parameters) // 2  # total number of layers
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters

    def update_parameters_with_momentum(self, grads, v, beta, learning_rate):

        parameters = self.parameters
        L = len(parameters) // 2 # number of layers in the neural networks

        # Momentum update for each parameter
        for l in range(L):
            # compute velocities
            v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*grads["db" + str(l+1)]
            # update parameters
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]

        return parameters, v

    def update_parameters_with_adam(self, grads, v, s, t, learning_rate,
                                    beta1, beta2,  epsilon):
        """
        Update parameters using Adam

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """

        parameters = self.parameters
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-(beta1**t))
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-(beta1**t))

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*np.square(grads["dW" + str(l+1)])
            s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*np.square(grads["db" + str(l+1)])

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-(beta2**t))
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-(beta2**t))

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*(v_corrected["dW" + str(l+1)]/np.sqrt(s_corrected["dW" + str(l+1)] + epsilon))
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*(v_corrected["db" + str(l+1)]/np.sqrt(s_corrected["db" + str(l+1)] + epsilon))

        return parameters, v, s

    def fit(self, X, Y, optimizer, mini_batch_size, num_epochs, learning_rate=0.0075, regularization=None, lamda=0,
            beta=0, beta1=0.9, beta2=0.999,  epsilon=1e-8, print_cost=True):
        """
        optimizer - can take values "gd", "momentum" or "adam"
        regularization - by default None, "L2" for L2 regularisation along with regularisation parameter lamda not 0
        """
        np.random.seed(1)
        L = len(self.layers_dims)
        m = X.shape[1]
        costs = []
        seed = 10  # for reshuffling minibatches
        t = 0  # initializing the counter for Adam update

        parameters = self.parameters
        # Initialize the optimizer
        if optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = initialize_adam(parameters)

        # Optimization loop
        for i in range(num_epochs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size=mini_batch_size, seed=0)
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                AL, caches = self.L_model_forward(minibatch_X)

                # Compute cost and add to the cost total
                cost_total += self.compute_cost(AL, minibatch_Y, lamda=lamda)

                # Backward propagation
                grads = self.L_model_backward(AL, minibatch_Y, caches, lamda=lamda)

                # Update parameters
                if optimizer == "gd":
                    parameters = self.update_parameters_with_gd(grads=grads, learning_rate=learning_rate)
                elif optimizer == "momentum":
                    parameters, v = self.update_parameters_with_momentum(grads=grads, v=v, beta=beta, learning_rate=learning_rate)
                elif optimizer == "adam":
                    t = t + 1  # Adam counter
                    parameters, v, s = self.update_parameters_with_adam(grads=grads, v=v, s=s,
                                                                   t=t, learning_rate=learning_rate, beta1=beta1, beta2=beta2,  epsilon=epsilon)
            cost_avg = cost_total / m

            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)



# Miscellaneous functions
def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        ### END CODE HERE ###

    return v

def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    ### END CODE HERE ###

    return v, s

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == 'softmax':
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def linear_backward(dZ, cache, lamda=0):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    lamda = regularization constant, default 0, else implements L2 reg with the passed lamda value

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    if lamda!=0:
        # add regularization term in dW
        dW = dW + (lamda/m)*W

    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, lamda=0):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lamda=lamda)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lamda=lamda)

    return dA_prev, dW, db

def sigmoid(Z):
    """Implements the sigmoid activation in numpy
    Arguments:
    Z -- numpy array of any shape
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation"""
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def softmax(Z):
    """Implement the Softmax function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A"; stored for computing the backward pass efficiently
    """
    t = np.exp(Z)
    A = t/(np.sum(t, axis=0))
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ

def load_data():
    train_dataset = h5py.File('data/train.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["image"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["label"][:]) # your train set labels

    test_dataset = h5py.File('data/test.hdf5', "r")
    test_set_x_orig = np.array(test_dataset["image"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["label"][:]) # your test set labels

    #classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig