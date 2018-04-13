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
from sklearn.model_selection import train_test_split
from utils import *
np.random.seed(42)

def RegressionMLP(X, y, layer_dims, X_test=None, y_test=None, lr=0.01, num_iters=1000, 
                  print_loss=True, add_del=False, print_add_del=False, del_threshold=0.03, 
                  prob_del=0.05, prob_add=0.05, max_hidden_size=1000, num_below_margin=5,
                  reg_param=0.):
    
    parameters, losses, test_losses = MLP(X, y, layer_dims, 'regression', X_test, y_test, lr, 
                             num_iters, print_loss, add_del, print_add_del, del_threshold, 
                             prob_del, prob_add, max_hidden_size, num_below_margin, reg_param)
    return parameters, losses, test_losses

def RegressionStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=1., 
                  prob_add=1., max_hidden_size=300, num_below_margin=5, reg_param=0.):
    
    parameters, losses, test_losses = StochasticMLP(X, y, layer_dims, 'regression', X_test, 
                                       y_test, optimizer, lr, batch_size,
                                       beta1, beta2, epsilon, num_epochs, print_loss, 
                                       add_del, print_add_del, del_threshold, prob_del, 
                                       prob_add, max_hidden_size, num_below_margin, reg_param)
    return parameters, losses, test_losses


if __name__ == '__main__':
    data_size = 1000
    num_features = 1
    
    X = 10.*np.random.rand(num_features,data_size)
    #y = 100.*(np.random.choice([1,-1],size=data_size)*np.random.rand(data_size))
    y = 10.*X[0,:]**2 - 3.
    y = y.reshape(1,-1)
    y += 100.*np.random.randn(1,y.shape[1])
    y = y.reshape(1,data_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T
    
    layer_dims = [X.shape[0],100, 1]
    parameters,_,_ = RegressionMLP(X_train, y_train, layer_dims, num_iters=1000,
                                   X_test=X_test, y_test=y_test,
                                   lr=0.1, print_loss=True)
    #parameters,_,_ = RegressionStochasticMLP(X_train, y_train, layer_dims, optimizer='adam',
    #                                         X_test=X_test,y_test=y_test,
    #                                         batch_size=128,lr=0.01,num_epochs=1000, 
    #                                         print_loss=True)
    print('training R^2 = %.3f' % score(X_train,y_train,parameters,'regression'))
    print('test R^2 = %.3f' % score(X_test,y_test,parameters,'regression'))
    
    # checking model works
    yhat = predict(X,y,parameters)
    plt.scatter(X[0,:],y[0,:],s=0.2)
    plt.scatter(X[0,:],yhat[0,:],color='red',s=0.2)
    plt.show()




