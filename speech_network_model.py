
# coding: utf-8

# # Constructing the Speech Network Model

# #### Import Modules for Building the Neural Network

# In[1]:

import numpy as np
import theano
import theano.tensor as T
import lasagne
import ctc_cost
import h5py
import random
import math


# ## Hyperparameters 

# In[2]:

#initial parameters
LEARNING_RATE = 0.001
NUM_EPOCHS=50
MOMENTUM=0.95
VALIDATION_SIZE=0.2

batch_size=100
feature_size=55
frames=777
num_phonomes=61


# ## Build the Bidirectional LSTM Network

# In[3]:

#Rebuilding the DBRNN
#Source: http://www.cs.toronto.edu/~graves/asru_2013.pdf

#input layer
input_layer=lasagne.layers.InputLayer(shape=(batch_size, feature_size, frames))
batchsize, feat, fm = input_layer.input_var.shape


#layer 1
fwd_layer_1=lasagne.layers.LSTMLayer(input_layer,num_units=256, backwards=False, learn_init=True)
bwd_layer_1=lasagne.layers.LSTMLayer(input_layer, num_units=256, backwards=True, learn_init=True)
recurrent_layer_1= lasagne.layers.ElemwiseSumLayer([fwd_layer_1,bwd_layer_1])

#layer 2
fwd_layer_2=lasagne.layers.LSTMLayer(recurrent_layer_1,num_units=256, backwards=False, learn_init=True)
bwd_layer_2=lasagne.layers.LSTMLayer(recurrent_layer_1, num_units=256, backwards=True, learn_init=True)
recurrent_layer_2= lasagne.layers.ElemwiseSumLayer([fwd_layer_2,bwd_layer_2])

#layer 3
fwd_layer_3=lasagne.layers.LSTMLayer(recurrent_layer_2,num_units=256, backwards=False, learn_init=True)
bwd_layer_3=lasagne.layers.LSTMLayer(recurrent_layer_2, num_units=256, backwards=True, learn_init=True)
recurrent_layer_3= lasagne.layers.ElemwiseSumLayer([fwd_layer_3,bwd_layer_3])

#layer 4
fwd_layer_4=lasagne.layers.LSTMLayer(recurrent_layer_3,num_units=256, backwards=False, learn_init=True)
bwd_layer_4=lasagne.layers.LSTMLayer(recurrent_layer_3, num_units=256, backwards=True, learn_init=True)
recurrent_layer_4= lasagne.layers.ElemwiseSumLayer([fwd_layer_4,bwd_layer_4])

#layer 5
fwd_layer_5=lasagne.layers.LSTMLayer(recurrent_layer_4,num_units=256, backwards=False, learn_init=True)
bwd_layer_5=lasagne.layers.LSTMLayer(recurrent_layer_4, num_units=256, backwards=True, learn_init=True)
recurrent_layer_5= lasagne.layers.ElemwiseSumLayer([fwd_layer_5,bwd_layer_5])

#connected layers
reshape_layer=lasagne.layers.ReshapeLayer(recurrent_layer_5,(-1,256))
densed_output_layer=lasagne.layers.DenseLayer(reshape_layer, num_units=num_phonomes, nonlinearity=lasagne.nonlinearities.identity)
output_reshape=lasagne.layers.ReshapeLayer(densed_output_layer, (batchsize,fm,num_phonomes))

#softmax of the connected layer
output_softmax=lasagne.layers.NonlinearityLayer(densed_output_layer, nonlinearity=lasagne.nonlinearities.softmax)
output_softmax_shp=lasagne.layers.ReshapeLayer(output_softmax, (batchsize, fm, num_phonomes))

output_lin_ctc=lasagne.layers.get_output(output_reshape)
network_output=lasagne.layers.get_output(output_softmax_shp)
all_params=lasagne.layers.get_all_params(recurrent_layer_5, trainable=True)


# ## Costs and Training Functions 

# In[4]:

# Cost functions
target_values = T.imatrix('target_output')
input_values  = T.imatrix()

### Gradients ###
# pseudo costs - ctc cross entropy b/n targets and linear output - used in training
pseudo_cost = ctc_cost.pseudo_cost(target_values, output_lin_ctc)
pseudo_cost_grad = T.grad(pseudo_cost.sum() / batchsize, all_params)
pseudo_cost = pseudo_cost.mean()

# true costs
cost = ctc_cost.cost(target_values, network_output)
cost = cost.mean()

# Compute SGD updates for training
print("Computing updates ...")
updates = lasagne.updates.rmsprop(pseudo_cost_grad, all_params, LEARNING_RATE)

# Theano functions for training and computing cost
print("Compiling functions ...")
train = theano.function([input_layer.input_var, target_values], [cost, pseudo_cost, network_output], updates=updates)
validate = theano.function([input_layer.input_var, target_values], [cost, network_output]) 
predict  = theano.function([input_layer.input_var], network_output)

#theano.printing.pydotprint(predict, outfile="prediction_graph.png", var_with_name_simple=True, compact=True)  


# ## Loading Data in Batches

# ### Extracting Audio File Names

# In[ ]:

max_frame_size=0
audio_names=[]


with h5py.File('timit_files/train_audio.h5', 'r') as h5:
    with open('audio_key.txt','r') as f:
        for line in f:
            line=line.rstrip()
            audio_names.append(line)
            cur=h5[line].shape[1]
            #print cur
            if cur>max_frame_size:
                max_frame_size=cur
                
number_of_audio_files=len(audio_names)

valid_size=int(math.floor(number_of_audio_files*VALIDATION_SIZE))
train_size=int(number_of_audio_files-valid_size)

#construct training set
training_set=random.sample(audio_names,train_size)

#construct validation set
validation_set=random.sample(audio_names,valid_size)


# In[6]:

'''
Insert Unique Audio Key to get output of Scaled MFCC, and Phenome

'''
def load_next_audio(audio_key):
    mfcc=None
    shape=None
    phn=None
    
    with h5py.File('timit_files/train_audio.h5', 'r') as h5:
        mfcc=h5[audio_key][:]
        mfcc=mfcc.astype(np.float32)
    with h5py.File('timit_files/train_phenome.h5', 'r') as h5:
        phn=h5[audio_key][:]

    return mfcc, phn


# In[7]:

#Example
mfcc,phn=load_next_audio('MBGT0_SX261')
print mfcc.shape


# In[8]:

'''
Input a audio key list that you want to pass.
Uses a global variable index to go through the list for each instance that the function is called
Returns the next Audio key
'''
audio_ctr=-1

def getNextAudioKey(audio_list):
    global audio_ctr
    audio_ctr+=1
    return audio_list[audio_ctr]


# In[9]:

test=getNextAudioKey(audio_names)
print test


# ## Training Network 

# In[10]:

print("Training network ...")
split_ratio = 0.8*batch_size


for epoch in range(NUM_EPOCHS):
    
    tlosses = []
    validation_losses = []
    plosses = []
    probabilities = []
    
    for batch in range(batch_size):
        key=getNextAudioKey(audio_names)
        mfcc,phn=load_next_audio(key)
        mfcc=mfcc.reshape(1,mfcc.shape[0],mfcc.shape[1])
        print phn
        if batch < split_ratio:
            loss, ploss, probs = train(mfcc,phn)
            tlosses.append(loss)
            plosses.append(ploss)
        else:
            loss, probs = validate(mfcc,phn)
            y_pred = np.argmax(probs, axis=-1)
            vlosses.append(loss)
            probabilities.append(probs)    
            
        print("Batch {0}/{1}, loss:{2:.6}, ploss:{3:.6}".format(batch,num_batches_train,loss,ploss))

