# Numpy implementation of self-selecting MLP for binary classification
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: May 2018
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.signal import lfilter
import time
from ss_perf_utils import *

np.random.seed(2)

def train_numpy(X,y,layer_dims,num_iters,lr=0.01,add_del=False):
    sigmoid = lambda z : 1./(1+np.exp(-z))
    
    din,dh,dout = tuple(layer_dims)
    m = X.shape[1]
    delta,prob,epsilon,max_hidden_size,tau = init_add_del()
    losses = []
    num_neurons = []
    
    W1 = np.random.randn(dh,din)
    b1 = np.random.randn(dh,1)
    W2 = np.random.randn(dout,dh)
    b2 = np.random.randn(dout,1)
    
    for t in range(num_iters):
        # Forwardprop
        Z1 = np.dot(W1,X)+b1
        A = Z1.clip(min=0) # relu
        Z2 = np.dot(W2,A)+b2
        yhat = sigmoid(Z2).clip(1e-6,1.-1e-6)
    
        loss = 1./m*(-np.dot(y,np.log(yhat).T)-np.dot(1-y,np.log(1-yhat).T))
        loss = loss.squeeze().item()
        losses.append(loss)
    
        # Backprop
        dyhat = -(np.divide(y,yhat) - np.divide(1-y, 1-yhat))
        dZ2 = dyhat*sigmoid(Z2)*(1-sigmoid(Z2))
        dW2 = 1./m*np.dot(dZ2,A.T)
        db2 = 1./m*np.sum(dZ2,1,keepdims=True)
        dA = np.dot(W2.T,dZ2)
        dZ1 = dA
        dZ1[Z1 < 0] = 0
        dW1 = 1./m*np.dot(dZ1,X.T)
        db1 = 1./m*np.sum(dZ1,1,keepdims=True)
    
        # gradient descent
        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

        if add_del and t>tau:
            W1,b1,W2,b2 = delete_neurons_numpy(W1,b1,W2,b2,delta,prob)
            W1,b1,W2,b2 = add_neurons_numpy(W1,b1,W2,b2,losses,epsilon,delta,
                                              max_hidden_size,tau,prob)
        num_neurons.append(b1.shape[0])

        if t % max(1,num_iters // 20) == 0:
            print('loss after iteration %i: %f' % (t, losses[-1]))
            if add_del:
                print('# neurons after iteration %i: %d' % (t, num_neurons[-1]))
    
    return losses,num_neurons


if __name__ == '__main__':
    num_iters = 10000
    num_samples = 1000
    num_features = 2
    num_hidden = 1
    num_classes = 1
    lr = 0.1
    layer_dims = [num_features,num_hidden,num_classes]
    
    X,y,x1,x2 = gen_data(samples=num_samples,var=0.01)
    X = X.T
    y = y.reshape(1,-1)
    
    tin = time.clock()
    losses,num_neurons = train_numpy(X,y,layer_dims,num_iters,lr=lr,add_del=True)
    tout = time.clock()
    tdiff = tout-tin
    print('time = %f' % tdiff)
    
    #losses = np.array(losses)
    filt_neurons = lfilter([1.0/50]*50,1,num_neurons)
    filt_neurons[filt_neurons<1] = num_hidden
    
    plt.plot(losses,color='blue')
    plt.title('Loss')
    plt.show()
    
    plt.plot(filt_neurons,color='green')
    plt.title('# Neurons')
    plt.show()    
    
    #plot_model(model,x1,x2)
    #print score(model,X,y)
