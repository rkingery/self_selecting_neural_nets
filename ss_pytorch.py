# PyTorch implementation of self-selecting MLP for binary classification
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: May 2018
# License: BSD 3 clause

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from ss_functions import *
np.random.seed(2)
torch.manual_seed(42)

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

def plot_model(model,x1,x2):
    xx, yy = np.mgrid[-3.5:3.5:.01, -3.5:3.5:.01]
    XX = np.c_[xx.ravel(), yy.ravel()]

    yhat = model(torch.FloatTensor(XX)).data.numpy().reshape(xx.shape)
    
    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, yhat, levels=[.5])
    
    ax.scatter(x1[:,0],x1[:,1],marker='.',c='red',label='y=0')
    ax.scatter(x2[:,0],x2[:,1],marker='.',c='blue',label='y=1')
    
    ax.set(aspect="equal",
           xlim=(-3.5, 3.5), ylim=(-3.5, 3.5),
           xlabel="$X_1$", ylabel="$X_2$")
    ax.legend(loc='upper right')
    plt.show()

#class Model(nn.Module):    
#    def __init__(self,num_features,num_hidden=1):
#        super(Model,self).__init__()
#        self.fc1 = nn.Linear(num_features, num_hidden)
#        self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(num_hidden, 1)
#        self.out = nn.Sigmoid()
#        
#    def forward(self, x):
#        x = self.fc1(x)
#        x = self.relu1(x)
#        x = self.fc2(x)
#        yhat = self.out(x)
#        return yhat

def score(model,X,y):
    yhat = model(X)
    pred = np.zeros(y.shape)
    for i in range(yhat.shape[0]):
        if yhat[i] > 0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    y = y.numpy().flatten()
    acc = accuracy_score(y,pred)
    return acc

def init_add_del():
    delta = 0.01
    prob = 1.
    epsilon = 1e-5
    max_hidden_size = 100
    tau = 50
    return delta,prob,epsilon,max_hidden_size,tau

#def train(model,X,y,lr,add_del=False):
#    delta,prob,epsilon,max_hidden_size,tau = init_add_del()
#    losses = []
#    num_neurons = []
#    #opt = optim.SGD(model.parameters(), lr=lr, momentum=0)
#    criterion = nn.BCELoss()
#    for t in range(num_iters):
#        #opt.zero_grad()
#        yhat = model(X)
#        loss = criterion(yhat,y)
#        losses.append(loss)
#        loss.backward()
#        with torch.no_grad():
#            #opt.step()
#            for param in model.parameters():
#                param -= lr * param.grad
#            if add_del and t>=tau:
#                model = delete_neurons_pytorch(model,delta,prob)
#                #model = add_neurons_pytorch(model,losses,epsilon,max_hidden_size,
#                #                            tau,prob,delta)
#            num_neurons.append(model.fc1.out_features)
#        if t % max(1,num_iters // 20) == 0:
#            print('loss after iteration %i: %f' % (t, losses[-1]))
#            if add_del:
#                print('# neurons after iteration %i: %d' % (t, num_neurons[-1]))
#    return losses,num_neurons

def train(X,y,layer_dims,num_iters,lr=0.01,add_del=False):
    dtype = torch.float
    #device = torch.device("cpu")
    sigmoid = lambda z : 1./(1+torch.exp(-z))
    
    din,dh,dout = tuple(layer_dims)
    m = X.shape[1]
    delta,prob,epsilon,max_hidden_size,tau = init_add_del()
    losses = []
    num_neurons = []
    
    W1 = torch.randn(dh, din, dtype=dtype, requires_grad=False)
    b1 = torch.randn(dh, 1, dtype=dtype, requires_grad=False)
    W2 = torch.randn(dout, dh, dtype=dtype, requires_grad=False)
    b2 = torch.randn(dout, 1, dtype=dtype, requires_grad=False)
    
    for t in range(num_iters):
        # Forwardprop
        Z1 = torch.mm(W1,X)+b1
        A = Z1.clamp(min=0) # relu
        Z2 = torch.mm(W2,A)+b2
        yhat = sigmoid(Z2).clamp(1e-6,1.-1e-6)
    
        criterion = nn.BCELoss()
        loss = criterion(yhat,y)
        loss = loss.squeeze_().item()
        losses.append(loss)
    
        # Backprop
        dyhat = -(torch.div(y,yhat) - torch.div(1-y, 1-yhat))
        dZ2 = dyhat*sigmoid(Z2)*(1-sigmoid(Z2))
        dW2 = 1./m*torch.mm(dZ2,A.t())
        db2 = 1./m*torch.sum(dZ2,1,keepdim=True)
        dA = torch.mm(W2.t(),dZ2)
        dZ1 = dA
        dZ1[Z1 < 0] = 0
        dW1 = 1./m*torch.mm(dZ1,X.t())
        db1 = 1./m*torch.sum(dZ1,1,keepdim=True)
    
        # gradient descent
        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

        if add_del and t>tau:
            W1,b1,W2,b2 = delete_neurons_pytorch(W1,b1,W2,b2,delta,prob)
            W1,b1,W2,b2 = add_neurons_pytorch(W1,b1,W2,b2,losses,epsilon,
                                                  delta,max_hidden_size,tau,prob)
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
    
    X,y,x1,x2 = gen_data(size=num_samples,var=0.01)
    X = torch.FloatTensor(X).t()
    y = torch.FloatTensor(y).reshape(1,-1)
    
    layer_dims = [X.shape[0],num_hidden,1]
    losses,num_neurons = train(X,y,layer_dims,num_iters,lr=lr,add_del=True)
    
    plt.plot(losses)
    plt.show()
    
    #plt.plot(num_neurons)
    #plt.show()    
    
    #plot_model(model,x1,x2)
    #print score(model,X,y)
