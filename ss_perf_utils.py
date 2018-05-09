import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.signal import lfilter
import time

def delete_neurons_pytorch(W1,b1,W2,b2,delta,prob):
    """
    For PyTorch models, deletes neurons with small outgoing weights from layer
    
    Arguments:
    W1 -- weight matrix into hidden layer
    b1 -- bias vector into hidden layer
    W2 -- weight matrix into output layer
    b2 -- bias vector into output layer
    delta -- threshold for deletion of neurons
    prob -- probability of a neuron below threshold being deleted
    
    Returns:
    updated W1, b1, W2, b2
    """
    
    W_out = W2   
    hidden_size = W_out.shape[1]
    
    norms = torch.sum(torch.abs(W_out),dim=0)
    max_out = torch.max(norms)
    selected = (norms == norms) # initialize all True == keep all neurons
    
    for j in range(hidden_size):
        norm = norms[j]
        if (norm < delta*max_out) and (torch.rand(1) < prob):
            # remove neuron j with probability prob
            selected[j] = 0
    
    if torch.sum(selected) == 0:
        # don't want ALL neurons in layer deleted or training will crash
        # keep neuron with largest outgoing weights if this occurs
        selected[torch.argmax(norms)] = 1
             
    W1 = W1[selected,:]
    b1 = b1[selected,:]
    W2 = W2[:,selected]
        
    return W1,b1,W2,b2


def add_neurons_pytorch(W1,b1,W2,b2,losses,epsilon,delta,max_hidden_size,tau,
                        prob,device):
    """
    For PyTorch models, adds neuron to bottom of layer if loss is stalling
    
    Arguments:
    W1 -- weight matrix into hidden layer
    b1 -- bias vector into hidden layer
    W2 -- weight matrix into output layer
    b2 -- bias vector into output layer
    epsilon -- range loss function must deviate to not be flagged as stalling
    delta -- threshold for deletion of neurons
    max_hidden_size -- max size allowable for the hidden layer
    tau -- window size to check for stalling in loss function
    prob -- probability of a neuron below threshold being deleted
    
    Returns:
    updated W1, b1, W2, b2
    """
    
    W_in = W1
    b_in = b1
    W_out = W2   
    hidden_size = W_out.shape[1]
    losses = torch.FloatTensor(losses)
    
    if hidden_size >= max_hidden_size:
        return W1,b1,W2,b2
    
    max_loss = torch.max(losses)
    filt_losses = losses#lfilter([1.0/5]*5,1,losses) # filter noise with FIR filter
    losses = filt_losses[-tau:]  # keep only losses in window t-tau,...,t
    upper = torch.mean(losses) + epsilon*max_loss
    lower = torch.mean(losses) - epsilon*max_loss
    num_out_of_window = (losses < lower) + (losses > upper)

    if (torch.sum(num_out_of_window).item() == 0) and (torch.rand(1) < prob):
        # if losses in window are too similar, add neuron with probability prob
        ones = torch.ones(1,W_in.shape[1],device=device)#.cuda()
        new_W_in = torch.tensor(torch.normal(0,2.*delta*ones),device=device)#.cuda()
        new_b_in = torch.zeros(1,1,device=device)#.cuda()
        ones = torch.ones(W_out.shape[0],1,device=device)#.cuda()
        new_W_out = torch.tensor(torch.normal(0,5.*delta*ones),device=device)#.cuda()
        W_in = torch.cat((W_in, new_W_in), dim=0)
        b_in = torch.cat((b_in, new_b_in), dim=0)
        W_out = torch.cat((W_out, new_W_out), dim=1)    
    
    W1 = W_in
    b1 = b_in
    W2 = W_out
    
    return W1,b1,W2,b2

def gen_data(samples=1000,var=2.):
    """
    Generates sample data from a Gaussian mixture model with centroid variance var 
    """
    centers = 5
    M = [np.random.multivariate_normal(np.array([1,0]),.8*np.eye(2)) for i in range(centers)] +\
        [np.random.multivariate_normal(np.array([0,1]),.8*np.eye(2)) for i in range(centers)]
    
    X = np.zeros((samples,2))
    y = np.zeros((samples,))
    x1 = []
    x2 = []    
    for j in range(samples):
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

def init_add_del():
    delta = 0.01
    prob = 1.
    epsilon = 1e-5
    max_hidden_size = 100
    tau = 50
    return delta,prob,epsilon,max_hidden_size,tau

#def plot_model(model,x1,x2):
#    xx, yy = np.mgrid[-3.5:3.5:.01, -3.5:3.5:.01]
#    XX = np.c_[xx.ravel(), yy.ravel()]
#
#    yhat = model(torch.FloatTensor(XX)).data.numpy().reshape(xx.shape)
#    
#    f, ax = plt.subplots(figsize=(8, 6))
#    ax.contour(xx, yy, yhat, levels=[.5])
#    
#    ax.scatter(x1[:,0],x1[:,1],marker='.',c='red',label='y=0')
#    ax.scatter(x2[:,0],x2[:,1],marker='.',c='blue',label='y=1')
#    
#    ax.set(aspect="equal",
#           xlim=(-3.5, 3.5), ylim=(-3.5, 3.5),
#           xlabel="$X_1$", ylabel="$X_2$")
#    ax.legend(loc='upper right')
#    plt.show()

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

#def score(model,X,y):
#    yhat = model(X)
#    pred = np.zeros(y.shape)
#    for i in range(yhat.shape[0]):
#        if yhat[i] > 0.5:
#            pred[i] = 1
#        else:
#            pred[i] = 0
#    y = y.numpy().flatten()
#    acc = accuracy_score(y,pred)
#    return acc

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