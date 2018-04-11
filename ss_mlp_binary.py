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
    
def BinaryMLP(X, y, layers_dims, lr=0.01, num_iters=1000, print_loss=True, 
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=0.05, 
                  prob_add=0.05, max_hidden_size=1000, num_below_margin=5):
    
    parameters, losses = MLP(X, y, layers_dims, 'binary', lr, num_iters, 
                             print_loss, add_del, print_add_del, del_threshold, prob_del, 
                             prob_add, max_hidden_size, num_below_margin)
    return parameters, losses

def BinaryStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=1., 
                  prob_add=1., max_hidden_size=300, num_below_margin=5):
    
    parameters, losses, test_losses = StochasticMLP(X, y, layer_dims, 'binary', X_test, 
                                       y_test, optimizer, lr, batch_size,
                                       beta1, beta2, eps, num_epochs, print_loss, 
                                       add_del, print_add_del, del_threshold, 
                                       prob_del, prob_add, max_hidden_size, num_below_margin)
    return parameters, losses, test_losses

def predict(X, y, parameters):
    """
    Uses neural net parameters to predict labels for input data X
    
    Arguments:
    X -- dataset to predict labels for
    parameters -- parameters of the trained model
    
    Returns:
    preds -- predictioned binary labels for dataset X
    """   
    preds = _predict(X, y, parameters, 'binary')
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
    acc = _score(X, y, parameters, 'binary')
    return acc

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
    num_iters = 1000
    lr = 0.1
#    parameters,reg_loss = BinaryMLP(X, y, layer_dims, num_iters=num_iters, print_loss=True)
#    print('accuracy = %.3f' % score(X,y,parameters))
    #parameters,reg_loss = MLP(X, y, layer_dims, num_iters=5000, print_loss=True, print_add_del=True)
    #parameters = StochasticMLP(X, y, layer_dims, optimizer='adam', batch_size=128,
    #                  num_epochs=500, print_loss=True)
    #print('accuracy = %.3f' % score(X,y,parameters))

    parameters,_,reg_loss = BinaryStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
                                        num_epochs=num_iters, lr=lr, add_del=False, optimizer='sgd', 
                                        batch_size=1000, print_loss=True, print_add_del=False)
    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters))
    parameters,_,ad_loss = BinaryStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
                                       num_epochs=num_iters, lr=lr, add_del=True, optimizer='sgd', 
                                       batch_size=1000, print_loss=True, print_add_del=False)
    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters))

    xx = np.arange(1,num_iters+1)
    plt.plot(xx,ad_loss,color='blue',label='add/del')
    plt.plot(xx,reg_loss,color='red',label='regular')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Test Loss')
    plt.show()
    
