# Various utility functions for self-selecting neural net implementations
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import logsumexp
np.random.seed(42)

def add_del_neurons(parameters, print_add_del, itr, del_threshold=0.05, prob_del=0.05, 
                    prob_add=0.05, max_hidden_size=300, num_below_margin=1):
    """
    Deletes and/or adds hidden layer neurons at the end of each epoch
    Arguments:
        parameters --
        print_add_del --
        itr --
        del_threshold --
        prob_del --
        prob_add --
        max_hidden_size --
        num_below_margin --
       
    Returns:
        parameters --
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
        print('neuron deleted at iteration '+str(itr))
            
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
            new_Why = .5*del_threshold*new_Why / (1e-8 + np.sum(np.abs(new_Why))) #+ 0.05
            Why = np.append(Why, new_Why, axis=1)
            
            bh = np.append(bh, 0)
            bh = bh.reshape(bh.shape[0],1)
            
            # also need memory terms here if updating per mini-batch
            if print_add_del and Wxh.shape[0] > hidden_size:
               print('neuron added at iteration '+str(itr))
            
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

