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
from scipy.signal import lfilter
from utils import *
np.random.seed(42)

def RegressionMLP(X, y, layer_dims, X_test=None, y_test=None, lr=0.01, num_iters=1000, 
                  print_loss=True, add_del=False, print_add_del=False, 
                  reg_param=0.,delta=0.01, prob=1., epsilon=0.1, max_hidden_size=100, tau=50):
                  #del_threshold=0.03, prob_del=0.05, prob_add=0.05, max_hidden_size=300, num_below_margin=5):
    
    parameters, losses, test_losses, num_neurons = \
        MLP(X, y, layer_dims, 'regression', X_test, y_test, lr, num_iters, print_loss, 
            add_del, reg_param, delta,prob,epsilon,max_hidden_size,tau)
    return parameters, losses, test_losses, num_neurons

def RegressionStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, reg_param=0.,
                  delta=0.01, prob=0.5, epsilon=0.05, max_hidden_size=100, tau=30):
                  #del_threshold=0.03, prob_del=1., prob_add=1., max_hidden_size=300, num_below_margin=1):
    
    parameters, losses, test_losses = \
        StochasticMLP(X, y, layer_dims, 'regression', X_test, y_test, optimizer, lr, batch_size,
                  beta1, beta2, eps, num_epochs, print_loss, add_del, print_add_del, reg_param,
                  delta,prob,epsilon,max_hidden_size,tau)
    
    return parameters, losses, test_losses


if __name__ == '__main__':
    data_size = 1000
    num_features = 1
    
    X = 10.*np.random.rand(num_features,data_size)
    #y = 100.*(np.random.choice([1,-1],size=data_size)*np.random.rand(data_size))
    y = 10.*X[0,:]**2 - 3.
    y = y.reshape(1,-1)
    y += 10.*np.random.randn(1,y.shape[1])
    y = y.reshape(1,data_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T
    
    num_iters = 20000
    lr = 0.1
    
    layer_dims = [X.shape[0],1, 1]
    parameters,_,losses,num_neurons = \
    RegressionMLP(X_train, y_train, layer_dims, num_iters=num_iters,
                  X_test=X_test, y_test=y_test,
                  lr=0.1, print_loss=True, add_del=True)
#    parameters,_,_ = RegressionStochasticMLP(X_train, y_train, layer_dims, optimizer='sgd',
#                                             X_test=None,y_test=None,
#                                             batch_size=128,lr=0.01,num_epochs=5000, 
#                                             print_loss=True, add_del=True)
    print('training R^2 = %.3f' % score(X_train,y_train,parameters,'regression'))
    print('test R^2 = %.3f' % score(X_test,y_test,parameters,'regression'))
    
    #checking model works
    yhat = predict(X,parameters,'regression')
    plt.scatter(X[0,:],y[0,:],s=0.2)
    plt.scatter(X[0,:],yhat[0,:],color='red',s=0.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('$y=10 x^2 - 3$')
    plt.show()
    
    losses = np.array(losses)
    num_neurons = np.array(num_neurons)

    xx = np.linspace(0,num_iters,num=num_iters)+1
    plt.plot(xx,1e-5*np.max(num_neurons)*losses,color='blue',label='val loss')
    filt_neurons = lfilter([1.0/50]*50,1,num_neurons)
    plt.plot(xx,filt_neurons,color='green',label='neurons')
    plt.legend(loc='middle right')
    plt.xlabel('iteration')
    #plt.ylabel('loss')
    plt.title('$y=10 x^2 - 3$')
    plt.show()


