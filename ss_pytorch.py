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

class Model(nn.Module):    
    def __init__(self,num_features,num_hidden=1):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, 1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        yhat = self.out(x)
        return yhat

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

def train(model,X,y,opt,criterion):
    losses = []
    for t in range(num_iters):
        yhat = model(X)
        loss = criterion(yhat,y)
        losses.append(loss)
        if t % max(1,num_iters // 20) == 0:
            print t, loss.item()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            opt.step()
            #for param in model.parameters():
            #    param -= lr * param.grad
    return losses

if __name__ == '__main__':
    num_iters = 500
    num_samples = 1000
    num_features = 2
    num_hidden = 10
    num_classes = 1
    
    X,y,x1,x2 = gen_data(size=num_samples,var=0.01)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)#.reshape(-1,1))#.to(torch.int64)
    
    model = Model(num_features,num_hidden)
    loss = nn.BCELoss()
    lr = 0.1
    bs = X.shape[0]
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0)
    #opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    #scheduler = lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)
    
    losses = train(model,X,y,opt,loss)
    
    plt.plot(losses)
    plt.show()
    
    plot_model(model,x1,x2)
    print score(model,X,y)
