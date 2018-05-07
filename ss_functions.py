# File contains all self-selection functions to add/delete neurons
# Notation used mostly follows Andrew Ng's deeplearning.ai course
# Author: Ryan Kingery (rkinger@g.clemson.edu)
# Last Updated: April 2018
# License: BSD 3 clause

# Functions contained:
#   delete_neurons
#   add_neurons
#   delete_neurons_adam
#   add_neurons_adam
#   add_del_neurons_orig

import numpy as np
import torch
from scipy.signal import lfilter
#np.random.seed(42)


def delete_neurons(parameters,delta,prob):
    """
    Deletes neurons with small outgoing weights from layer
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    l = 1   # applying to layer 1, plan to extend to other layers later
    
    W_out = parameters['W'+str(l+1)]    
    hidden_size = W_out.shape[1]
    
    norms = np.sum(np.abs(W_out),axis=0)
    max_out = np.max(norms)
    selected = (norms == norms) # initialize all True == keep all neurons
    
    for j in range(hidden_size):
        norm = norms[j]
        if (norm < delta*max_out) and (np.random.rand() < prob):
            # remove neuron j with probability prob
            selected[j] = False
    
    if np.sum(selected) == 0:
        # don't want ALL neurons in layer deleted or training will crash
        # keep neuron with largest outgoing weights if this occurs
        selected[np.argmax(norms)] = True
                    
    parameters['W'+str(l)] = parameters['W'+str(l)][selected,:]
    parameters['b'+str(l)] = parameters['b'+str(l)][selected,:]
    parameters['W'+str(l+1)] = parameters['W'+str(l+1)][:,selected]
    
    return parameters


def add_neurons(parameters,losses,epsilon,delta,max_hidden_size,tau,prob):
    """
    Add neuron to bottom of layer if loss is stalling
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    l = 1   # applying to layer 1, plan to extend to other layers later
    
    W_in = parameters['W'+str(l)]
    b_in = parameters['b'+str(l)]
    W_out = parameters['W'+str(l+1)]    
    hidden_size = b_in.shape[0]
    
    if hidden_size >= max_hidden_size:
        return parameters
    
    max_loss = np.max(losses)
    filt_losses = lfilter([1.0/5]*5,1,losses) # filter noise with FIR filter
    losses = filt_losses[-tau:]  # keep only losses in window t-tau,...,t
    upper = np.mean(losses) + epsilon*max_loss
    lower = np.mean(losses) - epsilon*max_loss
    num_out_of_window = np.logical_or((losses < lower),(losses > upper))
    
    if (np.sum(num_out_of_window) == 0) and (np.random.rand() < prob):
        # if losses in window are too similar, add neuron with probability prob
        delta = 0.1#3.*delta
        new_W_in = np.random.normal(0,2.*delta,size=(1,W_in.shape[1]))
        new_b_in = np.zeros((1,1))
        new_W_out = np.random.normal(0,2.*delta,size=(W_out.shape[0],1))
        W_in = np.append(W_in, new_W_in, axis=0)
        b_in = np.append(b_in, new_b_in, axis=0)
        W_out = np.append(W_out, new_W_out, axis=1)    
    
    parameters['W'+str(l)] = W_in
    parameters['b'+str(l)] = b_in
    parameters['W'+str(l+1)] = W_out
    
    return parameters    


def delete_neurons_adam(parameters,m,v,delta,prob):
    """
    Deletes neurons with small outgoing weights from layer, for use with Adam
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    l = 1   # applying to layer 1, plan to extend to other layers later
    
    W_out = parameters['W'+str(l+1)]    
    hidden_size = W_out.shape[1]
    
    norms = np.sum(np.abs(W_out),axis=0)
    selected = (norms == norms) # initialize all True == keep all neurons
    
    for j in range(hidden_size):
        norm = norms[j]
        if (norm < delta) and (np.random.rand() < prob):
            # remove neuron j with probability prob
            selected[j] = False
    
    if np.sum(selected) == 0:
        # don't want ALL neurons in layer deleted or training will crash
        # keep neuron with largest outgoing weights if this occurs
        selected[np.argmax(norms)] = True
                    
    parameters['W'+str(l)] = parameters['W'+str(l)][selected,:]
    parameters['b'+str(l)] = parameters['b'+str(l)][selected,:]
    parameters['W'+str(l+1)] = parameters['W'+str(l+1)][:,selected]

    m['dW'+str(l)] = m['dW'+str(l)][selected,:]
    m['db'+str(l)] = m['db'+str(l)][selected,:]
    m['dW'+str(l+1)] = m['dW'+str(l+1)][:,selected]
    
    v['dW'+str(l)] = v['dW'+str(l)][selected,:]
    v['db'+str(l)] = v['db'+str(l)][selected,:]
    v['dW'+str(l+1)] = v['dW'+str(l+1)][:,selected]
    
    return parameters,m,v


def add_neurons_adam(parameters,m,v,losses,epsilon,max_hidden_size,tau,prob):
    """
    Add neuron to bottom of layer if loss is stalling, for use with Adam
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    l = 1   # applying to layer 1, plan to extend to other layers later
    
    W_in = parameters['W'+str(l)]
    b_in = parameters['b'+str(l)]
    W_out = parameters['W'+str(l+1)]    
    hidden_size = b_in.shape[0]
    
    mW_in = m['dW'+str(l)]
    mb_in = m['db'+str(l)]
    mW_out = m['dW'+str(l+1)]
    
    vW_in = v['dW'+str(l)]
    vb_in = v['db'+str(l)]
    vW_out = v['dW'+str(l+1)]

    if hidden_size >= max_hidden_size:
        return parameters,m,v
    
    #max_loss = np.max(losses)
    filt_losses = lfilter([1.0/5]*5,1,losses) # filter noise with FIR filter
    losses = filt_losses[-tau:]  # keep only losses in window t-tau,...,t
    upper = np.mean(losses) + epsilon#*max_loss
    lower = np.mean(losses) - epsilon#*max_loss
    num_out_of_window = np.logical_or((losses < lower),(losses > upper))
    
    if (np.sum(num_out_of_window) == 0) and (np.random.rand() < prob):
        # if losses in window are too similar, add neuron with probability prob
        W_in = np.append(W_in, .01*np.random.randn(1,W_in.shape[1]), axis=0)
        b_in = np.append(b_in, np.zeros((1,1)), axis=0)
        W_out = np.append(W_out, .01*np.random.randn(W_out.shape[0],1), axis=1)
        
        mW_in = np.append(mW_in, .01*np.ones((1,W_in.shape[1])), axis=0)
        mb_in = np.append(mb_in, .01*np.ones((1,1)), axis=0)
        mW_out = np.append(mW_out, .01*np.ones((W_out.shape[0],1)), axis=1)

        vW_in = np.append(vW_in, .01*np.ones((1,W_in.shape[1])), axis=0)
        vb_in = np.append(vb_in, .01*np.ones((1,1)), axis=0)
        vW_out = np.append(vW_out, .01*np.ones((W_out.shape[0],1)), axis=1)        
    
    parameters['W'+str(l)] = W_in
    parameters['b'+str(l)] = b_in
    parameters['W'+str(l+1)] = W_out
    
    m['dW'+str(l)] = mW_in
    m['db'+str(l)] = mb_in
    m['dW'+str(l+1)] = mW_out
    
    v['dW'+str(l)] = vW_in
    v['db'+str(l)] = vb_in
    v['dW'+str(l+1)] = vW_out
    
    return parameters,m,v


def add_del_neurons_orig(parameters, itr, del_threshold, prob_del, prob_add, 
                         max_hidden_size, num_below_margin, print_add_del=False):
    """
    Original add_del_neurons function, closely follows Miconi
    Deletes and/or adds hidden layer neurons at the end of each epoch
    Arguments:
        parameters -- dict of parameters (weights and biases)
        print_add_del -- prints if neuron added/deleted if True (boolean)
        itr -- iteration of training (positive int)
        del_threshold -- threshold value determining neural deletion (>0)
        prob_del -- probability of deleting neuron if below threshold (0,...,1)
        prob_add -- probability of adding neuron at each iteration (0,...,1)
        max_hidden_size -- preferred max size of hidden layer (>0)
        num_below_margin -- number of below-threshold neurons not deleted (>0)
       
    Returns:
        parameters -- new dict of parameters with neurons added/deleted
    """
    assert len(parameters) == 2+2, \
    'self-selecting MLP only works with 1 hidden layer currently'
    
    Wxh = parameters['W1']
    Why = parameters['W2']
    bh = parameters['b1']
    num_features = Wxh.shape[1]
    num_labels = Why.shape[0]
    normz = (np.sum(np.abs(Why), axis = 0)) *.5
    selected = (np.abs(normz) > del_threshold)
    hidden_size = Wxh.shape[0]
    
    # deleting neurons
    if np.sum(selected) < hidden_size - num_below_margin:
        deletable = np.where(selected==False)[0]
        np.random.shuffle(deletable)
        for xx in range(num_below_margin):
            selected[deletable[xx]] = True
        deletable = deletable[num_below_margin:]
        for x in deletable:
            if np.random.rand() > prob_del:
                selected[x] = True
    
    if print_add_del and np.sum(selected) < hidden_size:
        print('neuron deleted at iteration %d' % itr)
            
    hidden_size = np.sum(selected)
    
    Wxh = Wxh[selected,:]
    normz = normz[selected]
    Why = Why[:,selected]
    bh = bh[selected]
    #need memory terms if updated per mini-batch iter instead of per epoch
    
    # adding neurons
    if hidden_size < max_hidden_size-1:
        if ( np.sum(np.abs(normz) > del_threshold) ) > hidden_size - num_below_margin \
            and ( np.random.rand() < prob_add ) or ( np.random.rand() < 1e-4 ):
            Wxh = np.append(Wxh, 0.01*np.random.randn(1,num_features), axis=0)
            
            new_Why = np.random.randn(num_labels,1)
            new_Why = .5*del_threshold*new_Why / (1e-8 + np.sum(np.abs(new_Why)))# + 0.05
            Why = np.append(Why, new_Why, axis=1)
            
            bh = np.append(bh, 0)
            bh = bh.reshape(bh.shape[0],1)
            
            # also need memory terms here if updating per mini-batch
            if print_add_del and Wxh.shape[0] > hidden_size:
               print('neuron added at iteration %d' % itr)
            
            hidden_size += 1
          
    parameters['W1'] = Wxh
    parameters['W2'] = Why
    parameters['b1'] = bh
    #self.hidden_layer_sizes[0] = hidden_size
    return parameters

def delete_neurons_pytorch(W1,b1,W2,b2,delta,prob):
    """
    For PyTorch models, deletes neurons with small outgoing weights from layer
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


def add_neurons_pytorch(W1,b1,W2,b2,losses,epsilon,delta,max_hidden_size,tau,prob):
    """
    For PyTorch models, adds neuron to bottom of layer if loss is stalling
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
    #print torch.sum(num_out_of_window).item()
    if (torch.sum(num_out_of_window).item() == 0) and (torch.rand(1) < prob):
        # if losses in window are too similar, add neuron with probability prob
        ones = torch.ones(1,W_in.shape[1])
        new_W_in = torch.normal(0,2.*delta*ones)
        new_b_in = torch.zeros(1,1)
        ones = torch.ones(W_out.shape[0],1)
        new_W_out = torch.normal(0,5.*delta*ones)
        W_in = torch.cat((W_in, new_W_in), dim=0)
        b_in = torch.cat((b_in, new_b_in), dim=0)
        W_out = torch.cat((W_out, new_W_out), dim=1)    
    
    W1 = W_in
    b1 = b_in
    W2 = W_out
    
    return W1,b1,W2,b2