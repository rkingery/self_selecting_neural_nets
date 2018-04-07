# Various utility functions for self-selecting neural net implementations
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

# Functions contained:
#   add_del_neurons
#   sigmoid
#   relu
#   softmax
#   initialize_parameters
#   gradient_descent
#
#
#

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.special import logsumexp
np.random.seed(42)

def add_del_neurons(parameters, print_add_del, itr, del_threshold, prob_del, 
                    prob_add, max_hidden_size, num_below_margin):
    """
    Deletes and/or adds hidden layer neurons at the end of each epoch
    Arguments:
        parameters -- dict of parameters (weights and biases)
        print_add_del -- prints if neuron added/deleted if True (boolean)
        itr -- iteration of training (positive int)
        del_threshold -- threshold value determining neural deletion (>0)
        prob_del -- probability of deleting neuron if below threshold (0,...,1)
        prob_add -- probability of adding neuron at each iteration (0,...,1)
        max_hidden_size -- preferred max size of hidden layer (>0)
        num_below_margin -- number of below-threshold neurons not deleted (>0)
       
    Returns:
        parameters -- new dict of parameters with neurons added/deleted
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    
    Wxh = parameters['W1']
    Why = parameters['W2']
    bh = parameters['b1']
    num_features = Wxh.shape[1]
    num_labels = Why.shape[0]
    normz = (np.sum(np.abs(Why), axis = 0)) *.5
    selected = (np.abs(normz) > del_threshold)
    hidden_size = Wxh.shape[0]
    
    # deleting neurons
    if np.sum(selected) < hidden_size - num_below_margin:
        deletable = np.where(selected==False)[0]
        np.random.shuffle(deletable)
        for xx in range(num_below_margin):
            selected[deletable[xx]] = True
        deletable = deletable[num_below_margin:]
        for x in deletable:
            if np.random.rand() > prob_del:
                selected[x] = True
    
    if print_add_del and np.sum(selected) < hidden_size:
        print('neuron deleted at iteration %d' % itr)
            
    hidden_size = np.sum(selected)
    
    Wxh = Wxh[selected,:]
    normz = normz[selected]
    Why = Why[:,selected]
    bh = bh[selected]
    #need memory terms if updated per mini-batch iter instead of per epoch
    
    # adding neurons
    if hidden_size < max_hidden_size-1:
        if ( np.sum(np.abs(normz) > del_threshold) ) > hidden_size - num_below_margin \
            and ( np.random.rand() < prob_add ) or ( np.random.rand() < 1e-4 ):
            Wxh = np.append(Wxh, 0.01*np.random.randn(1,num_features), axis=0)
            
            new_Why = np.random.randn(num_labels,1)
            new_Why = .5*del_threshold*new_Why / (1e-8 + np.sum(np.abs(new_Why))) + 0.05
            Why = np.append(Why, new_Why, axis=1)
            
            bh = np.append(bh, 0)
            bh = bh.reshape(bh.shape[0],1)
            
            # also need memory terms here if updating per mini-batch
            if print_add_del and Wxh.shape[0] > hidden_size:
               print('neuron added at iteration %d' % itr)
            
            hidden_size += 1
          
    parameters['W1'] = Wxh
    parameters['W2'] = Why
    parameters['b1'] = bh
    #self.hidden_layer_sizes[0] = hidden_size
    return parameters

def sigmoid(Z):
    """
    Implements the sigmoid function a = 1/(1+exp(-z))
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    output of sigmoid(z) function, same shape as Z
    """   
    return 1./(1+np.exp(-Z))

def relu(Z):
    """
    Implements the RELU function a=max(0,z)

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- output of relu(Z) function, same shape as Z
    """    
    A = np.maximum(0,Z)    
    return A

def softmax(Z):
    """
    Numerically stable implementation of the softmax function a = exp(z)/(sum(exp(z)))
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    output of softmax(z) function, same shape as Z
    """   
    logA = Z - logsumexp(Z,axis=0).reshape(1,Z.shape[1])
    return np.exp(logA)

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- list containing the dimensions of each layer in network
    
    Returns:
    parameters -- dict of parameters W1, b1, ..., WL, bL:
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1, L):
        # using He initialization
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
                                   np.sqrt(2. / layer_dims[l - 1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))        
    return parameters

def gradient_descent(parameters, grads, lr):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- dict containing network parameters 
    grads -- dict containing gradients backprop
    lr -- learning rate for gradient descent (default=0.001)
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = Wl 
                  parameters["b" + str(l)] = bl
    """

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        W = parameters['W' + str(l+1)]
        dW = grads['dW' + str(l+1)]
        b = parameters['b' + str(l+1)]
        db = grads['db' + str(l+1)]
        
        W = W - lr * dW        
        b = b - lr * db
        
        parameters['W' + str(l+1)] = W
        parameters['b' + str(l+1)] = b
                
    return parameters

def random_mini_batches(X, y, batch_size, seed):
    """
    Creates a list of random minibatches from (X, y)
    
    Arguments:
    X -- input data, of shape (num features, data size)
    y -- true labels, of shape (num labels, data size)
    batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_y)
    """
    
    np.random.seed(seed)
    
    m = X.shape[1]
    k = y.shape[0]
    mini_batches = []
    
    if batch_size == m:
        mini_batches = [(X,y)]
    else:
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_y = y[:, permutation].reshape((k,m))
    
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(np.floor(m*1./batch_size)) # number of batches in partition
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * batch_size : (k+1) * batch_size]
            mini_batch_y = shuffled_y[:, k * batch_size : (k+1) * batch_size]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size :]
            mini_batch_y = shuffled_y[:, num_complete_minibatches * batch_size :]
            mini_batch = (mini_batch_X, mini_batch_y)
            mini_batches.append(mini_batch)
    return mini_batches

def initialize_momentum(parameters):
    """
    Initializes momentum as a dict:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: array of zeros, same size as parameters
    Arguments:
    parameters -- dict of parameters
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    m -- dict of momentum values
                    m['dW' + str(l)] = momentum of dWl
                    m['db' + str(l)] = momentum of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    m = {}

    for l in range(L):
        m['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        m['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return m

def momentum(parameters, grads, m, beta, lr):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- dict of parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- dict of gradients for each parameter:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    m -- dict of current momentum values:
                    m['dW' + str(l)] = ...
                    m['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    lr -- the learning rate, scalar
    
    Returns:
    parameters -- dict of parameters 
    m -- dict of momentum values
    """

    L = len(parameters) // 2 # number of layers in the neural network
    
    for l in range(L):
        # compute momentum
        m["dW" + str(l+1)] = beta*m["dW" + str(l+1)] + (1-beta)*grads['dW' + str(l+1)]
        m["db" + str(l+1)] = beta*m["db" + str(l+1)] + (1-beta)*grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] -= lr*m["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= lr*m["db" + str(l+1)]
        
    return parameters, m

def initialize_adam(parameters) :
    """
    Initializes m and v as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    m -- dict containing exponentially weighted average of the gradient.
                    m["dW" + str(l)] = ...
                    m["db" + str(l)] = ...
    v -- dict containing exponentially weighted average of the squared gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural network
    m = {}
    v = {}

    for l in range(L):
        m["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        m["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return m, v

def adam(parameters, grads, m, v, t, lr, beta1, beta2, epsilon):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    m -- Adam variable, moving average of the first gradient, dict
    v -- Adam variable, moving average of the squared gradient, dict
    lr -- the learning rate, scalar
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    m -- Adam variable, moving average of the first gradient, dict
    v -- Adam variable, moving average of the squared gradient, dict
    """
    
    L = len(parameters) // 2         # number of layers in the neural network
    m_corrected = {}
    v_corrected = {}
    
    for l in range(L):
        # Moving average of the gradients
        m["dW" + str(l+1)] = beta1*m["dW" + str(l+1)] + (1-beta1)*grads['dW' + str(l+1)]
        m["db" + str(l+1)] = beta1*m["db" + str(l+1)] + (1-beta1)*grads['db' + str(l+1)]

        # Compute bias-corrected first moment estimate
        m_corrected["dW" + str(l+1)] = m["dW" + str(l+1)]#/(1-beta1**t)
        m_corrected["db" + str(l+1)] = m["db" + str(l+1)]#/(1-beta1**t)

        # Moving average of the squared gradients
        v["dW" + str(l+1)] = beta2*v["dW" + str(l+1)] + (1-beta2)*np.power(grads['dW' + str(l+1)],2)
        v["db" + str(l+1)] = beta2*v["db" + str(l+1)] + (1-beta2)*np.power(grads['db' + str(l+1)],2)

        # Compute bias-corrected second raw moment estimate
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta2**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta2**t)

        # Update parameters
        parameters["W" + str(l+1)] -= lr*np.divide(m_corrected["dW" + str(l+1)],
                  np.sqrt(v_corrected["dW" + str(l+1)])+epsilon)
        parameters["b" + str(l+1)] -= lr*np.divide(m_corrected["db" + str(l+1)],
                  np.sqrt(v_corrected["db" + str(l+1)])+epsilon)

    return parameters, m, v


