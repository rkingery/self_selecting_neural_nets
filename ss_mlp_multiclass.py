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
from sklearn.datasets import fetch_mldata
from utils import *
np.random.seed(42)

def MulticlassMLP(X, y, layers_dims, X_test=None, y_test=None, lr=0.01, num_iters=1000, 
                  print_loss=True, add_del=False, print_add_del=False, del_threshold=0.03, 
                  prob_del=0.05, prob_add=0.05, max_hidden_size=1000, num_below_margin=5,
                  reg_param=0.001):
    
    parameters, losses, test_losses = MLP(X, y, layers_dims, 'multiclass', X_test, y_test, lr, 
                             num_iters, print_loss, add_del, print_add_del,del_threshold, 
                             prob_del, prob_add, max_hidden_size, num_below_margin,reg_param)
    return parameters, losses, test_losses

def MulticlassStochasticMLP(X, y, layer_dims, X_test=None, y_test=None, optimizer='sgd', 
                  lr=0.0007, batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                  num_epochs=10000, print_loss=True,
                  add_del=False, print_add_del=False, del_threshold=0.03, prob_del=1., 
                  prob_add=1., max_hidden_size=300, num_below_margin=5, reg_param=0.001):
    
    parameters, losses, test_losses = StochasticMLP(X, y, layer_dims, 'multiclass', X_test, 
                                       y_test, optimizer, lr, batch_size, beta1, beta2, 
                                       epsilon, num_epochs, print_loss, add_del, print_add_del, 
                                       del_threshold, prob_del, prob_add, max_hidden_size, 
                                       num_below_margin, reg_param)
    return parameters, losses, test_losses

def predict(X, parameters):
    """
    Uses neural net parameters to predict labels for input data X
    
    Arguments:
    X -- dataset to predict labels for
    parameters -- parameters of the trained model
    
    Returns:
    preds -- predictioned binary labels for dataset X
    """   
    preds = _predict(X, parameters, 'multiclass')
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
    acc = _score(X, y, parameters, 'multiclass')
    return acc

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
    
    down_sample = 500
    X = X[:,:down_sample]
    y = y[:,:down_sample]
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    num_iters = 100
    lr = 0.01
    num_features = X_train.shape[0]
    num_classes = y_train.shape[0]
    layer_dims = [num_features, 100, num_classes]
    parameters,_,_ = MulticlassMLP(X_train,y_train, layer_dims, num_iters=num_iters,
                                   X_test=X_test,y_test=y_test,
                                   lr=lr, print_loss=True, add_del=False)
    #parameters,losses,test_losses = MulticlassStochasticMLP(X_train, y_train, layer_dims, 
    #                                              optimizer='adam', batch_size=128, 
    #                                              num_epochs=num_iters, print_loss=True)      
    print('training accuracy = %.3f' % score(X_train,y_train,parameters))
    print('test accuracy = %.3f' % score(X_test,y_test,parameters))
    
    
#    parameters,_,reg_loss = MulticlassStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
#                                        num_epochs=num_iters, lr=lr, add_del=False, optimizer='sgd', 
#                                        batch_size=128, print_loss=True, print_add_del=False)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters))
#    parameters,_,ad_loss = MulticlassStochasticMLP(X_train, y_train, layer_dims, X_test=X_test, y_test=y_test, 
#                                       num_epochs=num_iters, lr=lr, add_del=True, optimizer='sgd', 
#                                       batch_size=128, print_loss=True, print_add_del=False)
#    print('train accuracy = %.3f' % score(X_train,y_train,parameters))
#    print('test accuracy = %.3f' % score(X_test,y_test,parameters))
#
#    xx = np.arange(1,num_iters+1)
#    plt.plot(xx,ad_loss,color='blue',label='add/del')
#    plt.plot(xx,reg_loss,color='red',label='regular')
#    plt.legend(loc='upper right')
#    plt.xlabel('iteration')
#    plt.ylabel('loss')
#    plt.title('Test Loss')
#    plt.show()



