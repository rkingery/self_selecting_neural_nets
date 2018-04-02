# Implements a self-selecting multilayer perceptron for scalar regression
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import *
np.random.seed(42)

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
    A = Z
    inputs['Z'+str(L)] = Z
    inputs['A'+str(L)] = A
    yhat = A
    return yhat,inputs

def compute_loss(yhat, y):
    """
    Compute mean squared loss over dataset

    Arguments:
    activations -- dictionary of all activations from forward propagation
    y -- true response vector (real-valued), shape (1, num examples)

    Returns:
    loss -- mean squared loss
    """   
    m = y.shape[1]
    loss = 1./(2*m)*np.sum((y-yhat)**2)   
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
    
    m = y.shape[1]
    dyhat = -1./m * (y-yhat)
    
    Z = inputs['Z'+str(L)]
    A_prev = inputs['A'+str(L-1)]
    W = parameters['W'+str(L)]
    m = A_prev.shape[1]
    
    dZ = dyhat#*np.ones(yhat.shape)
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

def MLP(X, y, layers_dims, lr=0.01, num_iters=1000, print_loss=True, print_add_del=False,
        del_threshold=0.05, prob_del=0.05, prob_add=0.05, max_hidden_size=1000, num_below_margin=1):
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
        parameters = add_del_neurons(parameters,print_add_del,i,del_threshold, 
                                     prob_del,prob_add,max_hidden_size,num_below_margin)
                
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
    
    yhat,inputs = forwardprop(X, parameters)     
    return yhat
    
def score(X, y, parameters):
    """
    Calculates R^2 value on inputs X, responses y
    
    Arguments:
    X -- dataset to predict labels for
    y -- true labels for X
    parameters -- parameters of the trained model
    
    Returns:
    score -- returns R^2 value
    """  
    
    yhat = predict(X,y,parameters)    
    ssr = np.sum((y-yhat)**2)
    sst = np.sum((y-np.mean(y,axis=1))**2)
    score = 1.-ssr/sst   
    return score


if __name__ == '__main__':
    data_size = 100
    num_features = 1
    
    X = 10.*np.random.rand(num_features,data_size)
    #y = 100.*(np.random.choice([1,-1],size=data_size)*np.random.rand(data_size))
    y = 10.*X[0,:]**2 - 3.
    y = y.reshape(1,data_size)
    
    layers_dims = [X.shape[0],100, 1]
    parameters = MLP(X, y, layers_dims, num_iters=1000, lr=0.01, print_loss=True, print_add_del=True)
    print('R^2 = %.3f' % score(X,y,parameters))
    
    # checking model works
    yhat = predict(X,y,parameters)
    plt.scatter(X[0,:],y[0,:])
    plt.scatter(X[0,:],yhat[0,:],color='red')
    plt.show()




