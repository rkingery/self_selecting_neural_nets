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
from scipy.signal import lfilter
import time
from utils import *
np.random.seed(2)
    
def BinaryMLP(X, y, layer_dims, X_test=None, y_test=None, lr=0.01, num_iters=1000, 
                  print_loss=True, add_del=False, reg_param=0., delta=0.1, prob=1., 
                  epsilon=1e-4, max_hidden_size=100, tau=50):
                  #del_threshold=0.03, prob_del=0.05, prob_add=0.05, max_hidden_size=300, num_below_margin=5):
    
    parameters, losses, test_losses, num_neurons = \
        MLP(X, y, layer_dims, 'binary', X_test, y_test, lr, num_iters, print_loss, add_del, 
        reg_param, delta,prob,epsilon,max_hidden_size,tau)
    return parameters, losses, test_losses, num_neurons

def BinaryStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.01, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, 
                  num_epochs=1000, print_loss=True, add_del=False, reg_param=0.,
                  delta=0.03, prob=.5, epsilon=.001, max_hidden_size=100, tau=30):
                  #del_threshold=0.03, prob_del=1., prob_add=1., max_hidden_size=300, num_below_margin=1):
    
    parameters, losses, test_losses = \
        StochasticMLP(X, y, layer_dims, 'binary', X_test, y_test, optimizer, lr, batch_size,
                  beta1, beta2, eps, num_epochs, print_loss, add_del, reg_param,
                  delta,prob,epsilon,max_hidden_size,tau)
    
    return parameters, losses, test_losses

def gen_data(size=1000,var=2.):
    centers = 5
    M = [np.random.multivariate_normal(np.array([1,0]),.8*np.eye(2)) for i in range(centers)] +\
        [np.random.multivariate_normal(np.array([0,1]),.8*np.eye(2)) for i in range(centers)]
    
    X = np.zeros((size,2))
    y = np.zeros((size,))
    x1 = []
    x2 = []    
    for j in range(size):
        i = np.random.randint(2*centers)
        m = M[i]
        X[j,:] = np.random.multivariate_normal(np.array(m),var*np.eye(2)/centers)
        if i<centers:
            y[j] = 0
            x1 += [X[j,:]]
        else:
            y[j] = 1
            x2 += [X[j,:]]
    x1 = np.array(x1).reshape(len(x1),2)
    x2 = np.array(x2).reshape(len(x2),2)
    return X,y,x1,x2

def plot_model(parameters,x1,x2):
    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    XX = np.c_[xx.ravel(), yy.ravel()].T

    yhat = predict(XX,parameters,'binary').reshape(xx.shape)
    
    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, yhat, levels=[.5])#, cmap="Greys", vmin=0, vmax=.6)
    
    ax.scatter(x1[:,0],x1[:,1],marker='.',c='red',label='y=0')
    ax.scatter(x2[:,0],x2[:,1],marker='.',c='blue',label='y=1')
    
    ax.set(aspect="equal",
           xlim=(-3, 3), ylim=(-3, 3),
           xlabel="$X_1$", ylabel="$X_2$")
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    data_size = 1000
    num_features = 2
    
    #X = np.random.rand(num_features,data_size)
    #y = np.random.randint(0,2,data_size).reshape(1,data_size)
    X,y,x1,x2 = gen_data(size=data_size,var=0.01)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T.reshape(1,-1)
    y_test = y_test.T.reshape(1,-1)
    
    layer_dims = [X_train.shape[0], 1, 1]
    num_iters = 10000
    lr = 0.1
    #bs = X_train.shape[1] / 16
    
    tin = time.clock()
    parameters,_,ad_loss,num_neurons = \
        BinaryMLP(X_train, y_train, layer_dims, X_test=X_test,
                  y_test=y_test, num_iters=num_iters, add_del=True, 
                  print_loss=True, lr=lr)
    tout = time.clock()
    tdiff = tout-tin
    print('time = %f'% tdiff)
    
    print('training accuracy = %.3f' % score(X_train,y_train,parameters,'binary'))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'binary'))
    plot_model(parameters,x1,x2)
    
#    layer_dims = [X_train.shape[0], 10, 1]
#    parameters,_,reg_loss,_ = \
#    BinaryMLP(X_train, y_train, layer_dims, X_test=X_test,
#              y_test=y_test, num_iters=num_iters, add_del=False, 
#              print_loss=True, lr=lr)
#    print('training accuracy = %.3f' % score(X_train,y_train,parameters,'binary'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'binary'))
#    plot_model(parameters,x1,x2)

    xx = np.linspace(1,num_iters+1,num=num_iters)
    plt.plot(xx,2.*np.max(num_neurons)*np.array(ad_loss),color='blue',label='val loss')
    filt_neurons = lfilter([1.0/50]*50,1,num_neurons)
    plt.plot(xx,filt_neurons,color='green',label='neurons')
    plt.legend(loc='center right')
    plt.xlabel('iteration')
    #plt.ylabel('loss')
    plt.title('Gaussian Mixtures')
    plt.show()

#    parameters,_,ad_loss = BinaryStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
#                                        num_epochs=num_iters, lr=lr, add_del=True, optimizer='adam', 
#                                        batch_size=bs, print_loss=True)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters,'binary'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'binary'))
#    plot_model(parameters,x1,x2)
#    
#    parameters,_,reg_loss = BinaryStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
#                                       num_epochs=num_iters, lr=lr, add_del=False, optimizer='adam', 
#                                       batch_size=bs, print_loss=True)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters,'binary'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'binary'))

#    xx = np.linspace(1,num_iters+1,num=100)
#    plt.plot(xx,ad_loss,color='blue',label='add/del')
#    plt.plot(xx,reg_loss,color='red',label='regular')
#    plt.legend(loc='upper right')
#    plt.xlabel('iteration')
#    plt.ylabel('loss')
#    plt.title('Test Loss')
#    plt.show()
    
    #ps = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5]
