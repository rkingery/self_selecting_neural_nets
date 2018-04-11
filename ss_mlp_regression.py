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

def RegressionMLP(X, y, layer_dims, lr=0.01, num_iters=1000, print_loss=True, 
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=0.05, 
                  prob_add=0.05, max_hidden_size=1000, num_below_margin=5):
    
    parameters, losses = MLP(X, y, layer_dims, 'regression', lr, num_iters, 
                             print_loss, add_del, print_add_del, del_threshold, prob_del, 
                             prob_add, max_hidden_size, num_below_margin)
    return parameters, losses

def RegressionStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=1., 
                  prob_add=1., max_hidden_size=300, num_below_margin=5):
    
    parameters, losses, test_losses = StochasticMLP(X, y, layer_dims, 'regression', X_test, 
                                       y_test, optimizer, lr, batch_size,
                                       beta1, beta2, eps, num_epochs, print_loss, 
                                       add_del, print_add_del, del_threshold, 
                                       prob_del, prob_add, max_hidden_size, num_below_margin)
    return parameters, losses, test_losses

def predict(X, y, parameters):
    preds = _predict(X, y, parameters, 'regression')
    return preds

def score(X, y, parameters):
    acc = _score(X, y, parameters, 'regression')
    return acc


if __name__ == '__main__':
    data_size = 1000
    num_features = 1
    
    X = 10.*np.random.rand(num_features,data_size)
    #y = 100.*(np.random.choice([1,-1],size=data_size)*np.random.rand(data_size))
    y = 10.*X[0,:]**2 - 3.
    y = y.reshape(1,data_size)
    
    layer_dims = [X.shape[0],10, 1]
    parameters,losses = RegressionMLP(X, y, layer_dims, num_iters=1000, lr=0.1, print_loss=True)
    #parameters,losses,test_losses = RegressionStochasticMLP(X, y, layer_dims, optimizer='adam', batch_size=128,
    #                   lr=0.01,num_epochs=1000, print_loss=True)
    print('R^2 = %.3f' % score(X,y,parameters))
    
    # checking model works
    yhat = predict(X,y,parameters)
    plt.scatter(X[0,:],y[0,:])
    plt.scatter(X[0,:],yhat[0,:],color='red')
    plt.show()




