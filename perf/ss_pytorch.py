# PyTorch implementation of self-selecting MLP for binary classification
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: May 2018
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.signal import lfilter
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from ss_perf_utils import *

np.random.seed(2)
torch.manual_seed(2)

global device,dtype
#device = torch.device('cuda:0')
device = torch.device('cpu')
dtype = torch.float

def train_pytorch(X,y,layer_dims,num_iters,lr=0.01,add_del=False):
    sigmoid = lambda z : 1./(1+torch.exp(-z))
    
    din,dh,dout = tuple(layer_dims)
    m = X.shape[1]
    delta,prob,epsilon,max_hidden_size,tau = init_add_del()
    losses = []
    num_neurons = []
    
    print 'X',X.shape
    print 'y',y.shape
    
    W1 = torch.randn(dh, din, dtype=dtype, requires_grad=False, device=device)
    print 'W1',W1.shape
    b1 = torch.randn(dh, 1, dtype=dtype, requires_grad=False, device=device)
    print 'b1',b1.shape
    W2 = torch.randn(dout, dh, dtype=dtype, requires_grad=False, device=device)
    print 'W2',W2.shape
    b2 = torch.randn(dout, 1, dtype=dtype, requires_grad=False, device=device)
    print 'b2',b2.shape
    
    for t in range(num_iters):
        # Forwardprop
        Z1 = torch.mm(W1,X)+b1
        print 'Z1',Z1.shape
        A = Z1.clamp(min=0) # relu
        print 'A',Z1.shape
        Z2 = torch.mm(W2,A)+b2
        print 'z2',Z2.shape
        yhat = sigmoid(Z2).clamp(1e-6,1.-1e-6)
        print 'yhat',yhat.shape
    
        criterion = nn.BCELoss()
        loss = criterion(yhat,y)
        loss = loss.squeeze_().item()
        losses.append(loss)
    
        # Backprop
        dyhat = -(torch.div(y,yhat) - torch.div(1-y, 1-yhat))
        print 'dyhat',dyhat.shape
        dZ2 = dyhat*sigmoid(Z2)*(1-sigmoid(Z2))
        print 'dZ2',dZ2.shape
        dW2 = 1./m*torch.mm(dZ2,A.t())
        print 'dW2',W2.shape
        db2 = 1./m*torch.sum(dZ2,1,keepdim=True)
        print 'db2',db2.shape
        dA = torch.mm(W2.t(),dZ2)
        print 'dA',dA.shape
        dZ1 = dA
        dZ1[Z1 < 0] = 0
        print 'dZ1',dZ1.shape
        dW1 = 1./m*torch.mm(dZ1,X.t())
        print 'dW1',dW1.shape
        db1 = 1./m*torch.sum(dZ1,1,keepdim=True)
        print 'db1',db1.shape
    
        # gradient descent
        W1 -= lr*dW1
        b1 -= lr*db1
        W2 -= lr*dW2
        b2 -= lr*db2

        if add_del and t>tau:
            W1,b1,W2,b2 = delete_neurons_pytorch(W1,b1,W2,b2,delta,prob)
            W1,b1,W2,b2 = add_neurons_pytorch(W1,b1,W2,b2,losses,epsilon,delta,
                                              max_hidden_size,tau,prob,device)
        num_neurons.append(b1.shape[0])

        if t % max(1,num_iters // 20) == 0:
            print('loss after iteration %i: %f' % (t, losses[-1]))
            if add_del:
                print('# neurons after iteration %i: %d' % (t, num_neurons[-1]))
    
    return losses,num_neurons


if __name__ == '__main__':
    num_iters = 1
    num_samples = 1000
    num_features = 2
    num_hidden = 4
    num_classes = 1
    lr = 0.1
    layer_dims = [num_features,num_hidden,num_classes]
    
    X,y,x1,x2 = gen_data(samples=num_samples,var=0.01)
    X = torch.tensor(X,device=device,dtype=dtype).t()
    y = torch.tensor(y,device=device,dtype=dtype).reshape(1,-1)
    
    tin = time.clock()
    losses,num_neurons = train_pytorch(X,y,layer_dims,num_iters,lr=lr,add_del=True)
    tout = time.clock()
    tdiff = tout-tin
    print('time = %f' % tdiff)
    
    losses = np.array(losses)
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
