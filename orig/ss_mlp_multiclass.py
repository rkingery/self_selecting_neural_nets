# Implements a self-selecting multilayer perceptron for multiclass classification
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from scipy.signal import lfilter
from utils import *
np.random.seed(42)

def MulticlassMLP(X, y, layer_dims, X_test=None, y_test=None, lr=0.01, num_iters=1000, 
                  print_loss=False, add_del=False, reg_param=0.,delta=0.01, prob=1., 
                  epsilon=1e-3, max_hidden_size=200, tau=50):
                  #del_threshold=0.03, prob_del=0.05, prob_add=0.05, max_hidden_size=300, num_below_margin=5):
    
    parameters, losses, test_losses, num_neurons = \
        MLP(X, y, layer_dims, 'multiclass', X_test, y_test, lr, num_iters, print_loss, add_del, 
            reg_param, delta,prob,epsilon,max_hidden_size,tau)
    return parameters, losses, test_losses, num_neurons

def MulticlassStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, eps=1e-8, print_loss=False,
                  num_epochs=10000, add_del=False, print_add_del=False, reg_param=0.,
                  delta=0.01, prob=0.5, epsilon=0.05, max_hidden_size=100, tau=30):
                  #del_threshold=0.03, prob_del=1., prob_add=1., max_hidden_size=300, num_below_margin=1):
    
    parameters, losses, test_losses = \
        StochasticMLP(X, y, layer_dims, 'multiclass', X_test, y_test, optimizer, lr, batch_size,
                  beta1, beta2, eps, num_epochs, print_loss, add_del, reg_param,
                  delta,prob,epsilon,max_hidden_size,tau)
    
    return parameters, losses, test_losses

if __name__ == '__main__':
#    data_size = 7
#    num_features = 10
#    num_classes = 3
#    
#    X_train = 10.*np.random.rand(num_features,data_size)
#    y_train = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1],[1,0,0]]).T

    mnist = fetch_mldata('MNIST original', data_home=os.getcwd())    
    X = mnist.data.astype(np.float32) / 255.
    y_orig = mnist.target
    # one-hot encode the labels y_orig: i=0,...,9 --> [0,...,1,...,0]
    y = pd.get_dummies(y_orig).values.astype(np.float32)
    
#    pca = PCA(n_components=324)
#    pca.fit(X)
#    X_pca = pca.transform(X)
    
    X,y = shuffle(X,y)
    
    down_sample = 5000
    X_ds = X[:down_sample,:]
    y_ds = y[:down_sample,:]
    
    X_train,X_test,y_train,y_test = train_test_split(X_ds,y_ds,test_size=0.2)
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    num_iters = 10000
    lr = 0.01
    #bs = 128
    num_features = X_train.shape[0]
    num_classes = y_train.shape[0]
    layer_dims = [num_features, 1, num_classes]
    parameters,_,ad_loss,num_neurons = \
    MulticlassMLP(X_train, y_train, layer_dims, X_test=X_test,
                  y_test=y_test, num_iters=num_iters, add_del=True, 
                  print_loss=True, lr=lr)
    print('training accuracy = %.3f' % score(X_train,y_train,parameters,'multiclass'))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'multiclass'))
#    
#    #layer_dims = [num_features, 1, num_classes]
#    parameters,_,reg_loss = MulticlassMLP(X_train,y_train, layer_dims, num_iters=num_iters,
#                                   X_test=X_test,y_test=y_test,
#                                   lr=lr, print_loss=False, add_del=False)     
#    print('training accuracy = %.3f' % score(X_train,y_train,parameters,'multiclass'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'multiclass'))
    
    
#    parameters,adam_loss,_ = MulticlassStochasticMLP(X_train, y_train, layer_dims, X_test=None, y_test=None, 
#                                        num_epochs=num_iters, lr=lr, add_del=False, optimizer='adam', 
#                                        batch_size=bs, print_loss=False)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters,'multiclass'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'multiclass'))
#    parameters,sgd_loss,_ = MulticlassStochasticMLP(X_train, y_train, layer_dims, X_test=None, y_test=None, 
#                                       num_epochs=num_iters, lr=lr, add_del=False, optimizer='sgd', 
#                                       batch_size=bs, print_loss=False)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters,'multiclass'))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters,'multiclass'))

    ad_loss = np.array(ad_loss)
    num_neurons = np.array(num_neurons)

    xx = np.linspace(0,num_iters,num=num_iters)+1
    plt.plot(xx,.5*np.max(num_neurons)*ad_loss,color='blue',label='val loss')
    filt_neurons = lfilter([1.0/50]*50,1,num_neurons)
    plt.plot(xx,filt_neurons,color='green',label='neurons')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    #plt.ylabel('loss')
    plt.title('MNIST')
    plt.show()



