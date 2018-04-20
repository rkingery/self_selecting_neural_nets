# Various utility functions for self-selecting neural net implementations
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

# Functions contained:
#   sigmoid
#   tanh
#   relu
#   softmax
#   initialize_parameters
#   forwardprop
#   compute_loss
#   backprop
#   gradient_descent
#   random_mini_batches
#   initialize_momentum
#   momentum
#   initialize_adam
#   adam
#   MLP
#   StochasticMLP
#   predict
#   score

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from ss_functions import *
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

def tanh(Z):
    """
    Implements the hyperbolic tangent function a = tanh(z)
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    output of tanh(z) function, same shape as Z
    """   
    return np.tanh(Z)

def relu(Z):
    """
    Implements the RELU function a = max(0,z)

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- output of relu(Z) function, same shape as Z
    """        
    return np.maximum(0.,Z)

def softmax(Z):
    """
    Numerically stable implementation of the softmax function a = exp(z)/sum(exp(z))
    
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
    L = len(layer_dims)-1            # number of layers in the network
    for l in range(1, L+1):
        # using He initialization
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
                                   np.sqrt(2. / layer_dims[l - 1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))        
    return parameters

def forwardprop(X, parameters, problem_type):
    """
    Implements forward propagation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters function
    problem_type -- binary classification, regression, multiclass classification
                    (binary, regression, multiclass)
    
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
    if problem_type == 'regression':
        A = Z
    elif problem_type == 'binary':
        A = sigmoid(Z)
        A = np.clip(A,1e-8,1.-1e-8)     # clip to prevent loss from blowing up
    elif problem_type == 'multiclass':
        A = softmax(Z)
        A = np.clip(A,1e-8,1.-1e-8)     # clip to prevent loss from blowing up
    inputs['Z'+str(L)] = Z
    inputs['A'+str(L)] = A
    yhat = A
    return yhat,inputs

def flatten_weights(parameters):
    L = len(parameters) // 2
    w = np.array([])
    for l in range(1,L+1):
        w = np.append(w,parameters['W'+str(l)].flatten())
    return w

def compute_loss(yhat, y, parameters, reg_param, problem_type):
    """
    Compute average loss over dataset

    Arguments:
    activations -- dictionary of all activations from forward propagation
    y -- true "label" vector (e.g. 0 if non-cat, 1 if cat), shape (1, num examples)
    problem_type -- binary classification, regression, multiclass classification
                    (binary, regression, multiclass)

    Returns:
    loss -- cross-entropy loss
    """   
    m = y.shape[1]
    # calculate base loss
    if problem_type == 'regression':
        loss = 1./(2*m)*np.sum((y-yhat)**2)
    elif problem_type == 'binary':
        loss = 1./m*(-np.dot(y,np.log(yhat).T)-np.dot(1-y, np.log(1-yhat).T))
    elif problem_type == 'multiclass':
        loss = -1./m*np.sum(np.sum(y*np.log(yhat),axis=0))
    loss = np.squeeze(loss)      # turns [[17]] into 17).
    
    # add L1 regularization term
    w = flatten_weights(parameters)
    loss += 1./m*reg_param*np.sum(np.abs(w))
    
    return loss

def backprop(yhat, y, inputs, parameters, problem_type):
    """
    Implements backward propagation
    
    Arguments:
    yhat -- probability vector, output of the forward propagation
    y -- true "label" vector (e.g. 0 if non-cat, 1 if cat)
    inputs -- dict of inputs outputted from forward propagation:
    parameters -- dict of parameter weights and biases
    problem_type -- binary classification, regression, multiclass classification
                    (binary, regression, multiclass)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = d loss / dA
             grads["dW" + str(l)] = d loss / dW
             grads["db" + str(l)] = d loss / db
    """
    grads = {}
    L = len(parameters) // 2
    y = y.reshape(yhat.shape)
 
    Z = inputs['Z'+str(L)]
    A_prev = inputs['A'+str(L-1)]
    W = parameters['W'+str(L)]
    
    if problem_type == 'regression':
        m = y.shape[1]
        dyhat = -1./m * (y-yhat)          
        m = A_prev.shape[1]           
        dZ = dyhat
    elif problem_type == 'binary':
        dyhat = - (np.divide(y,yhat) - np.divide(1-y, 1-yhat))
        m = A_prev.shape[1]
        dZ = dyhat*sigmoid(Z)*(1-sigmoid(Z))
    elif problem_type == 'multiclass':
        m = A_prev.shape[1]  
        dZ = yhat-y
        
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

def gradient_descent(parameters, grads, lr, reg_param, data_size):
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
        parameters['W'+str(l+1)] -= lr*(grads['dW'+str(l+1)]) #+ 
                  #(1./data_size)*reg_param*np.sign(parameters['W'+str(l+1)]))
        parameters['b'+str(l+1)] -= lr*grads['db'+str(l+1)]                
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
    
    m = y.shape[1]
    k = y.shape[0]
    mini_batches = []
    
    #if batch_size >= m:
    #    mini_batches = [(X,y)]
    if True:#else:
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
    
    L = len(parameters) // 2 # number of layers
    m = {}

    for l in range(L):
        m['dW'+str(l+1)] = np.zeros(parameters['W'+str(l+1)].shape)
        m['db'+str(l+1)] = np.zeros(parameters['b'+str(l+1)].shape)
        
    return m

def momentum(parameters, grads, m, beta, lr, reg_param, data_size):
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

    L = len(parameters) // 2 # number of layers
    
    for l in range(L):
        # compute momentum
        m["dW" + str(l+1)] = beta*m["dW" + str(l+1)] + (1-beta)*grads['dW' + str(l+1)]
        m["db" + str(l+1)] = beta*m["db" + str(l+1)] + (1-beta)*grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] -= lr*(m["dW" + str(l+1)]) #+ 
                  #(1./data_size)*reg_param*np.sign(parameters['W'+str(l+1)]))
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
    
    L = len(parameters) // 2 # number of layers
    m = {}
    v = {}

    for l in range(L):
        m["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        m["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return m, v

def adam(parameters, grads, m, v, t, lr, beta1, beta2, epsilon, reg_param, data_size):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    m -- Adam variable, moving average of the first gradient moment, dict
    v -- Adam variable, moving average of the second gradient moment, dict
    lr -- the learning rate, scalar
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    m -- Adam variable, moving average of the first gradient moment, dict
    v -- Adam variable, moving average of the second gradient moment, dict
    """
    
    L = len(parameters) // 2         # number of layers
    m_corrected = {}
    v_corrected = {}
    
    for l in range(L):
        # Moving average of the gradients
        m["dW" + str(l+1)] = beta1*m["dW" + str(l+1)] + (1-beta1)*grads['dW' + str(l+1)]
        m["db" + str(l+1)] = beta1*m["db" + str(l+1)] + (1-beta1)*grads['db' + str(l+1)]
        #print np.all((m["dW" + str(l+1)] - grads['dW' + str(l+1)]) < 1e-5)
        # Compute bias-corrected first moment estimate
        m_corrected["dW" + str(l+1)] = m["dW" + str(l+1)]/(1-beta1**t)
        m_corrected["db" + str(l+1)] = m["db" + str(l+1)]/(1-beta1**t)

        # Moving average of the squared gradients
        v["dW" + str(l+1)] = beta2*v["dW" + str(l+1)] + (1-beta2)*np.power(grads['dW' + str(l+1)],2)
        v["db" + str(l+1)] = beta2*v["db" + str(l+1)] + (1-beta2)*np.power(grads['db' + str(l+1)],2)
        #print np.all((v["dW" + str(l+1)] - grads['dW' + str(l+1)]) < 1e-5)
        # Compute bias-corrected second raw moment estimate
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta2**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta2**t)

        # Update parameters
        parameters["W" + str(l+1)] -= lr*(np.divide(m_corrected["dW" + str(l+1)],
                  np.sqrt(v_corrected["dW" + str(l+1)])+epsilon)) #+ 
                  #(1./data_size)*reg_param*np.sign(parameters['W'+str(l+1)]))
        parameters["b" + str(l+1)] -= lr*np.divide(m_corrected["db" + str(l+1)],
                  np.sqrt(v_corrected["db" + str(l+1)])+epsilon)

    return parameters, m, v

def MLP(X, y, layer_dims, problem_type, X_test, y_test, lr, num_iters, print_loss, add_del, 
        reg_param, delta,prob,epsilon,max_hidden_size,tau):
        #del_threshold, prob_del, prob_add, max_hidden_size, num_below_margin):
    """
    Implements a L-layer multilayer perceptron (MLP)
    
    Arguments:
    X -- data, numpy array of shape (num data, num features)
    y -- true "label" vector (e.g. 0 if cat, 1 if non-cat), of shape (1, num data)
    layers_dims -- list containing input size and each layer size, length (num layers + 1).
    lr -- learning rate of the gradient descent update rule
    num_iters -- number of iterations of the optimization loop
    print_loss -- if True, it prints the cost every few steps
    
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
    losses = []                         # keep track of loss for plotting
    test_losses = []
    num_neurons = []
    
    # Parameters initialization.
    parameters = initialize_parameters(layer_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iters):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        yhat,inputs = forwardprop(X, parameters, problem_type)
        
        # Compute cost.
        loss = compute_loss(yhat, y, parameters, reg_param, problem_type)
    
        # Backward propagation.
        grads = backprop(yhat, y, inputs, parameters, problem_type)        
 
        # Update parameters.
        data_size = y.shape[1]
        parameters = gradient_descent(parameters, grads, lr, reg_param, data_size)

        # Add / delete neurons
        if add_del and i>tau:
            #parameters = add_del_neurons_orig(parameters,print_add_del,i,del_threshold, 
            #                             prob_del,prob_add,max_hidden_size,num_below_margin)
            parameters = delete_neurons(parameters,delta,prob)
            parameters = add_neurons(parameters,losses,epsilon,max_hidden_size,tau,prob)  
            
        if X_test is not None and y_test is not None:
            yhat_test,_ = forwardprop(X_test, parameters, problem_type)
            test_loss = compute_loss(yhat_test, y_test, parameters, reg_param, problem_type)

        if add_del:
            num_neuron = parameters['b1'].shape[0]

        # Print the cost every 100 training example
        num_prints = max(1,num_iters // 20)
        if print_loss and i % num_prints == 0:
            print('Loss after iteration %i: %f' % (i, loss))
            if add_del:
                print('Number of neurons %i: %d' % (i, num_neuron))
            if X_test is not None and y_test is not None:
                print ("Test loss after epoch %i: %f" %(i, test_loss))
                
        #num_losses = max(1,num_iters // 100)
        #if i % num_losses == 0:
        if True:
            losses.append(loss)
            if add_del:
                num_neurons.append(num_neuron)
            if X_test is not None and y_test is not None:
                    test_losses.append(test_loss)        
        
        #if i>0 and i%1000 == 0:
        #    lr = lr/(1+0.0*i)
        #    print('learning rate reduced to %f' % lr)
            
    # plot the cost
    if print_loss:
        plt.plot(losses,color='blue',label='train')
        if X_test is not None and y_test is not None:
            plt.plot(test_losses,color='red',label='test')
        plt.legend(loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('iterations')
        plt.title('Loss')
        plt.show()
        
    if add_del:
        plt.plot(num_neurons,color='green',label='# neurons')
        plt.ylabel('# neurons')
        plt.xlabel('epochs')
        plt.title('Number of neurons')
        plt.show()
        
    return parameters, losses, test_losses

def StochasticMLP(X, y, layer_dims, problem_type, X_test, y_test, optimizer, lr, batch_size,
                  beta1, beta2, eps, num_epochs, print_loss, add_del, reg_param,
                  delta,prob,epsilon,max_hidden_size,tau):
                  #del_threshold, prob_del, prob_add, max_hidden_size, num_below_margin):
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
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_loss -- True to print the loss every 1000 epochs

    Returns:
    parameters -- dict of parameters 
    """
    #X = X.T
    #y = y.T
    losses = []
    test_losses = []
    all_losses = []
    num_neurons = []
    t = 0                            # counter required for Adam update
    seed = 42
    
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)
    #num_neuron = parameters['b1'].shape[0]
    #num_neurons.append(num_neuron)

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
            yhat,inputs = forwardprop(minibatch_X, parameters, problem_type)

            # Compute cost
            loss = compute_loss(yhat, minibatch_y, parameters, reg_param, problem_type)

            # Backward propagation
            grads = backprop(yhat, minibatch_y, inputs, parameters, problem_type) 

            # Update parameters
            num_in_batch = minibatch_y.shape[1]
            if optimizer == "sgd":
                parameters = gradient_descent(parameters,grads,lr,reg_param,num_in_batch)
            elif optimizer == "momentum":
                parameters, m = momentum(parameters,grads,m,beta1,lr,reg_param,num_in_batch)
            elif optimizer == "adam":
                t += 1 # Adam counter
                parameters, m, v = adam(parameters,grads,m,v,t,lr,beta1,beta2,eps,
                                        reg_param,num_in_batch)
            all_losses.append(loss)
        
        # Add / delete neurons
        if add_del and i>tau:
            if optimizer == 'sgd':
                #parameters = add_del_neurons_orig(parameters,print_add_del,i,del_threshold, 
                #                         prob_del,prob_add,max_hidden_size,num_below_margin)
                parameters = delete_neurons(parameters,delta,prob)
                parameters = add_neurons(parameters,all_losses,epsilon,max_hidden_size,tau,prob)                
            if optimizer == 'adam':
                parameters,m,v = delete_neurons_adam(parameters,m,v,delta,prob)
                parameters,m,v = add_neurons_adam(parameters,m,v,all_losses,epsilon,max_hidden_size,tau,prob)
                #print len(add_neurons_adam(parameters,m,v,all_losses,epsilon,max_hidden_size,tau,prob))
            
        if X_test is not None and y_test is not None:
            minibatches = random_mini_batches(X_test, y_test, batch_size, seed)
            for minibatch in minibatches:

                # Select a minibatch
                minibatch_X, minibatch_y = minibatch
    
                # Forward propagation
                yhat_test,_ = forwardprop(minibatch_X, parameters, problem_type)
    
                # Compute cost
                test_loss = compute_loss(yhat_test, minibatch_y, parameters, reg_param, problem_type)
        
        if add_del:
            num_neuron = parameters['b1'].shape[0]
        
        # Print the every few losses
        num_prints = max(1,num_epochs // 20)
        if print_loss and i % num_prints == 0:
            print ("Training loss after epoch %i: %f" %(i, loss))
            if add_del:
                print ("Number of neurons %i: %d" %(i, num_neuron))
            if X_test is not None and y_test is not None:
                print ("Test loss after epoch %i: %f" %(i, test_loss))
                
        num_losses = max(1,num_epochs // 100)
        if i % num_losses == 0:
            losses.append(loss)
            if add_del:
                num_neurons.append(num_neuron)
            if X_test is not None and y_test is not None:
                    test_losses.append(test_loss)
                
    # plot the cost
    if print_loss:
        plt.plot(losses,color='blue',label='train')
        if X_test is not None and y_test is not None:
            plt.plot(test_losses,color='red',label='test')
        plt.legend(loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('Loss')
        plt.show()
        
    if add_del:
        plt.plot(num_neurons,color='green',label='# neurons')
        plt.ylabel('# neurons')
        plt.xlabel('epochs')
        plt.title('Number of neurons')
        plt.show()

    return parameters, losses, test_losses

def predict(X, parameters, problem_type):
    """
    Uses neural net parameters to predict labels for input data X
    
    Arguments:
    X -- dataset to predict labels for
    parameters -- parameters of the trained model
    
    Returns:
    preds -- predictioned labels for dataset X
    """   
    m = X.shape[1]
    yhat,_ = forwardprop(X, parameters, problem_type)
    preds = np.zeros(yhat.shape)
    if problem_type == 'regression':
        preds = yhat
    elif problem_type == 'binary':
        for i in range(0, yhat.shape[1]):
            if yhat[0,i] > 0.5:
                preds[0,i] = 1
            else:
                preds[0,i] = 0 
    elif problem_type == 'multiclass':
        max_idxs = np.argmax(yhat, axis=0)
        for i in range(m):
            imax = max_idxs[i]
            preds[imax,i] = 1
    return preds

def score(X, y, parameters, problem_type):
    """
    Calculates accuracy of neural net on inputs X, true labels y
    
    Arguments:
    X -- dataset to predict labels for
    y -- true labels for X
    parameters -- parameters of the trained model
    
    Returns:
    acc -- num correctly predict labels / num total labels (if classification)
           R^2 value (if regression)
    """
    m = X.shape[1]
    if problem_type == 'regression':
        preds = predict(X,parameters,problem_type)    
        ssr = np.sum((y-preds)**2)
        sst = np.sum((y-np.mean(y,axis=1))**2)
        acc = 1.-ssr/sst
    elif problem_type == 'binary':
        preds = predict(X,parameters,problem_type)
        acc = np.sum((preds == y)*1./m)
    elif problem_type == 'multiclass':
        yhat,_ = forwardprop(X, parameters, problem_type)
        acc = 1./m*np.count_nonzero(np.argmax(yhat, axis=0) == np.argmax(y, axis=0))
    return acc