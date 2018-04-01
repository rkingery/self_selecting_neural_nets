# Implements a self-selecting multilayer perceptron for binary classification
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(42)

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
                                   np.sqrt(2. / layers_dims[l - 1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))        
    return parameters

def forwardprop(X, parameters):
    """
    Implements forward propagation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters function
    
    Returns:
    yhat -- last post-activation value
    inputs -- dict of inputs containing:
                input data X
                every Z=W*A+b, for each l=1,...,L
                every A=activation(Z), for each l=1,...,L
    """
    inputs = {'A0':X}
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z = W.dot(A_prev) + b
        A = relu(Z)
        inputs['Z'+str(l)] = Z
        inputs['A'+str(l)] = A
    A_prev = A
    W = parameters['W'+str(L)]
    b = parameters['b'+str(L)]
    Z = W.dot(A_prev) + b
    A = sigmoid(Z)
    inputs['Z'+str(L)] = Z
    inputs['A'+str(L)] = A
    yhat = A
    return yhat,inputs

def compute_loss(yhat, y):
    """
    Compute average cross-entropy loss over dataset

    Arguments:
    activations -- dictionary of all activations from forward propagation
    y -- true "label" vector (e.g. 0 if non-cat, 1 if cat), shape (1, num examples)

    Returns:
    loss -- cross-entropy loss
    """   
    m = y.shape[1]
    loss = 1./m*(-np.dot(y,np.log(yhat).T) - np.dot(1-y, np.log(1-yhat).T))    
    loss = np.squeeze(loss)      # turns [[17]] into 17).    
    return loss

def backprop(yhat, y, inputs, parameters):
    """
    Implements backward propagation
    
    Arguments:
    yhat -- probability vector, output of the forward propagation
    y -- true "label" vector (e.g. 0 if non-cat, 1 if cat)
    inputs -- dict of inputs outputted from forward propagation:
    parameters -- dict of parameter weights and biases
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = d loss / dA
             grads["dW" + str(l)] = d loss / dW
             grads["db" + str(l)] = d loss / db
    """
    grads = {}
    L = len(parameters) // 2
    y = y.reshape(yhat.shape)
    
    dyhat = - (np.divide(y,yhat) - np.divide(1-y, 1-yhat))
    
    Z = inputs['Z'+str(L)]
    A_prev = inputs['A'+str(L-1)]
    W = parameters['W'+str(L)]
    m = A_prev.shape[1]
    
    dZ = dyhat*sigmoid(Z)*(1-sigmoid(Z))
    dA_prev = np.dot(W.T,dZ)
    dW = 1./m*np.dot(dZ,A_prev.T)
    db = 1./m*np.sum(dZ,axis=1,keepdims=True)

    grads['dA'+str(L-1)] = dA_prev
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    
    for l in reversed(range(1,L)):
        Z = inputs['Z'+str(l)]
        A_prev = inputs['A'+str(l-1)]
        W = parameters['W'+str(l)]
        m = A_prev.shape[1]
        
        dZ = np.array(dA_prev, copy=True)
        dZ[Z <= 0] = 0
        dA_prev = np.dot(W.T,dZ)
        dW = 1./m*np.dot(dZ,A_prev.T)
        db = 1./m*np.sum(dZ,axis=1,keepdims=True)

        grads['dA'+str(l-1)] = dA_prev
        grads['dW'+str(l)] = dW
        grads['db'+str(l)] = db
    
    return grads

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

def add_del_neurons(parameters, del_threshold=0.05, prob_del=0.05, prob_add=0.01, 
                    max_hidden_size=300, num_below_margin=1, print_add_del=False):
    """
    Deletes and/or adds hidden layer neurons at the end of each epoch
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
        print('neuron deleted')
            
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
            new_Why = .5*del_threshold*new_Why / (1e-8 + np.sum(np.abs(new_Why)))
            Why = np.append(Why, new_Why, axis=1)
            
            bh = np.append(bh, 0)
            bh = bh.reshape(bh.shape[0],1)
            
            # also need memory terms here if updating per mini-batch
            if print_add_del and Wxh.shape[0] > hidden_size:
               print('neuron added')
            
            hidden_size += 1
          
    parameters['W1'] = Wxh
    parameters['W2'] = Why
    parameters['b1'] = bh
    #self.hidden_layer_sizes[0] = hidden_size
    return parameters

def MLP(X, y, layers_dims, lr=0.01, num_iters=1000, print_loss=True):
    """
    Implements a L-layer multilayer perceptron (MLP)
    
    Arguments:
    X -- data, numpy array of shape (num data, num features)
    y -- true "label" vector (e.g. 0 if cat, 1 if non-cat), of shape (1, num data)
    layers_dims -- list containing input size and each layer size, length (num layers + 1).
    lr -- learning rate of the gradient descent update rule
    num_iters -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    losses = []                         # keep track of loss for plotting
    
    # Parameters initialization.
    parameters = initialize_parameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iters):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        yhat,inputs = forwardprop(X, parameters)
        
        # Compute cost.
        loss = compute_loss(yhat, y)
    
        # Backward propagation.
        grads = backprop(yhat, y, inputs, parameters)        
 
        # Update parameters.
        parameters = gradient_descent(parameters, grads, lr)
        
        # Add / delete neurons
        parameters = add_del_neurons(parameters)
                
        # Print the cost every 100 training example
        if print_loss and i % 100 == 0:
            print('Loss after iteration %i: %f' % (i, loss))
        if print_loss and i % 100 == 0:
            losses.append(loss)
            
    # plot the cost
    if print_loss:
        plt.plot(np.squeeze(losses))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('Training Loss')
        plt.show()
    
    return parameters

def predict(X, y, parameters):
    """
    Uses neural net parameters to predict labels for input data X
    
    Arguments:
    X -- dataset to predict labels for
    parameters -- parameters of the trained model
    
    Returns:
    preds -- predictioned binary labels for dataset X
    """   
    m = X.shape[1]
    preds = np.zeros((1,m))
    yhat,inputs = forwardprop(X, parameters)
    for i in range(0, yhat.shape[1]):
        if yhat[0,i] > 0.5:
            preds[0,i] = 1
        else:
            preds[0,i] = 0       
    return preds

def score(X, y, parameters):
    """
    Calculates accuracy of neural net on inputs X, true labels y
    
    Arguments:
    X -- dataset to predict labels for
    y -- true labels for X
    parameters -- parameters of the trained model
    
    Returns:
    acc -- num correctly predict labels / num total labels
    """  
    m = X.shape[1]
    preds = predict(X,y,parameters)
    acc = np.sum((preds == y)*1./m)
    return acc


if __name__ == '__main__':
    data_size = 1000
    num_features = 10
    
    X = np.random.rand(num_features,data_size)
    y = np.random.randint(0,2,data_size).reshape(1,data_size)
    
    layers_dims = [X.shape[0], 10, 1]
    parameters = MLP(X, y, layers_dims, num_iters=2000, print_loss=True)
    print 'accuracy = ',score(X,y,parameters)




