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
from sklearn.model_selection import train_test_split
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

def MLP(X, y, layers_dims, lr=0.01, num_iters=1000, print_loss=True, add_del=False, 
        print_add_del=False, del_threshold=0.03, prob_del=0.05, prob_add=0.05, 
        max_hidden_size=1000, num_below_margin=5):
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
        if add_del:
            parameters = add_del_neurons(parameters,print_add_del,i,del_threshold, 
                                         prob_del,prob_add,max_hidden_size,num_below_margin)
                
        # Print the cost every 100 training example
        if print_loss and i % 1000 == 0:
            print('Loss after iteration %i: %f' % (i, loss))
        if print_loss:# and i % 100 == 0:
            losses.append(loss)
            
    # plot the cost
    if print_loss:
        plt.plot(np.squeeze(losses))
        plt.ylabel('loss')
        plt.xlabel('iterations (per tens)')
        plt.title('Training Loss')
        plt.show()
    
    return parameters,losses

def StochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=1., 
                  prob_add=1., max_hidden_size=300, num_below_margin=5):
    """
    MLP which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (num features, data size)
    y -- true "label" vector, shape (num classes, data size)
    layer_dims -- list, containing the size of each layer
    lr -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    eps -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_loss -- True to print the loss every 1000 epochs

    Returns:
    parameters -- dict of parameters 
    """
    #X = X.T
    #y = y.T
    #L = len(layers_dims)             # number of layers in the neural network
    if len(y.shape) == 1:
        y = y.reshape(1,-1)
    losses = []
    test_losses = []
    t = 0                            # counter required for Adam update
    seed = 42
    
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Initialize the optimizer
    if optimizer == "sgd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        m = initialize_momentum(parameters)
    elif optimizer == "adam":
        m,v = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):        
        # Define the random minibatches
        seed += 1 # reshuffles the dataset differently after each epoch
        minibatches = random_mini_batches(X, y, batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            minibatch_X, minibatch_y = minibatch

            # Forward propagation
            yhat,inputs = forwardprop(minibatch_X, parameters)

            # Compute cost
            loss = compute_loss(yhat, minibatch_y)

            # Backward propagation
            grads = backprop(yhat, minibatch_y, inputs, parameters) 

            # Update parameters
            if optimizer == "sgd":
                parameters = gradient_descent(parameters,grads,lr)
            elif optimizer == "momentum":
                parameters, m = momentum(parameters,grads,m,beta1,lr)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, m, v = adam(parameters,grads,m,v,t,lr,beta1,beta2,eps)
                
        if X_test is not None and y_test is not None:
            minibatches = random_mini_batches(X_test, y_test, batch_size, seed)
            for minibatch in minibatches:

                # Select a minibatch
                minibatch_X, minibatch_y = minibatch
    
                # Forward propagation
                yhat,inputs = forwardprop(minibatch_X, parameters)
    
                # Compute cost
                test_loss = compute_loss(yhat, minibatch_y)
        
        # Add / delete neurons
        if add_del:
            parameters = add_del_neurons(parameters,print_add_del,i,del_threshold, 
                                         prob_del,prob_add,max_hidden_size,num_below_margin)
        
        # Print the cost every 1000 epoch
        if print_loss and i % 1000 == 0:
            print ("Training loss after epoch %i: %f" %(i, loss))
            print ("Test loss after epoch %i: %f" %(i, loss))
        if print_loss:# and i % 100 == 0:
            losses.append(loss)
            test_losses.append(test_loss)
                
    # plot the cost
    plt.plot(losses,color='blue',label='train')
    plt.plot(test_losses,color='red',label='test')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Training Loss')
    plt.show()

    return parameters,losses, test_losses

if __name__ == '__main__':
    data_size = 1000
    num_features = 10
    
    X = np.random.rand(num_features,data_size)
    y = np.random.randint(0,2,data_size).reshape(1,data_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T
    
    layer_dims = [X.shape[0], 10, 1]
    num_iters = 20000
    lr = 0.1
    #parameters,reg_loss = MLP(X, y, layer_dims, num_iters=num_iters, print_loss=True, print_add_del=True)
    #print('accuracy = %.3f' % score(X,y,parameters))
    #parameters,reg_loss = MLP(X, y, layer_dims, num_iters=5000, print_loss=True, print_add_del=True)
    #parameters = StochasticMLP(X, y, layer_dims, optimizer='adam', batch_size=128,
    #                  num_epochs=500, print_loss=True)
    #print('accuracy = %.3f' % score(X,y,parameters))

    parameters,_,reg_loss = StochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
                                        num_epochs=num_iters, lr=lr, add_del=False, optimizer='sgd', 
                                        batch_size=1000, print_loss=True, print_add_del=False)
    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters))
    parameters,_,ad_loss = StochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
                                       num_epochs=num_iters, lr=lr, add_del=True, optimizer='sgd', 
                                       batch_size=1000, print_loss=True, print_add_del=False)
    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters))

    xx = np.arange(1,num_iters+1)
    plt.plot(xx,ad_loss,color='blue',label='add/del')#,linestyle='none',marker='o')
    plt.plot(xx,reg_loss,color='red',label='regular')#,linestyle='none',marker='o')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Test Loss')
    plt.show()
    
