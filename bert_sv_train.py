# %%
# =============================================================================
# Libs
# =============================================================================
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
from bert.dataset import * #dataset class for BERT pretrainig
from model import * #BERT model
from bert.build_vocab import * #buliding vocabulary
from sv_dataset import * # dataset class for SV classification
from sv_trainer import * #trainer class for classifier
from sv_model import * # Classification model

# %% [markdown]
# Hyperparameters

# %%
# =============================================================================
# #Init
# =============================================================================

#-------------------------paths----------------------------
train_datapath = "data/train.tsv" #train dataset for train bert
valid_datapath = "data/valid.tsv" #valid dataset for BERT
test_datapath = "data/test.tsv" #test dataset for train bert
save_path = "classfn_saved_models/sv_trained"
bert_model_path = "bert_saved_models/bert_trained.ep30"
svc_model_path = "classfn_saved_models/sv_trained.ep10"
vocab_path = "wiki.vocab"

#------------------BERT hyperparameters-------------
embed_size = 128 #embedding size of BERT
inner_ff_size = embed_size * 4 #size of the inner FFN in BERT
n_layers = 6 #number of BERT layers
n_heads =  8 #number of attention heads
seq_len = 50 #maximum sequene length

#------------------Classifier hyperparameters------------------
batch_size = 64#number of batch_size
epochs = 11 #number of epochs
num_classes = 2 #number of classes

#optimizer arguments 
lr=1e-4
weight_decay = 1e-4
adam_betas = (.9, .999)
dropout = 0.1 #initially 0.1 changed to 0

#----------------------dataloader args-------------------
on_memory = True #Loading on memory: true or false
n_workers = 80#dataloader worker size
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size }

#------------------------------cuda device arguments
with_cuda= True #use cuda or not
cuda_devices = [5,6,7] #CUDA device ids

#print log frequency
log_freq =10

#index to find gradient of BERT
grad_index =-1 #-1 during training

#find the gradient of the classifier or not
find_grad = False

#input embedding given to BERT instead of input_ids
input_embedding = True

train_from_saved = False #if start training from a saved model
load_bert_pretrain = False # load pretrained bert

#_______________________________________________________________________________________________________________________

# Create vocabulary from train data
vocab = create_vocab(train_datapath)
len(vocab)

# --------------------create train, valid and test datasets
#create train dataset
train_dataset = SVDataset(train_datapath, vocab, seq_len)
valid_dataset = SVDataset(valid_datapath, vocab, seq_len)
test_dataset = SVDataset(test_datapath, vocab, seq_len)
vocab_size = len(train_dataset.vocab)
print(vocab_size)

item = train_dataset.__getitem__(0)

#------------------------- create dataloaders

train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **kwargs)
test_dataloader = torch.utils.data.DataLoader(test_dataset, **kwargs)

torch.manual_seed(0)

#---------BERT model--------------------
bert_model = BERT(n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size, 
        inner_ff_size=inner_ff_size,
        n_embeddings=vocab_size,
        input_embedding=input_embedding,
        seq_len=seq_len,
        dropout=dropout,
        grad_idx=grad_index) #create BERT object

if load_bert_pretrain: #if load pretrained model
    bert_model_state_dict = torch.load(bert_model_path) #load state dict
    bert_model.load_state_dict(bert_model_state_dict) # load state dict onto the model



svc_model_state_dict =None
if train_from_saved:
    svc_model_state_dict = torch.load(svc_model_path) 



# %%
# Create an instance  of SVTrainer
svn_trainer =   SVTrainer(bert_model,
                        num_classes, 
                        train_dataloader=train_dataloader, 
                        test_dataloader=valid_dataloader,
                        svc_model_state_dict=svc_model_state_dict,
                        find_grad=find_grad,
                        lr=lr, 
                        betas=adam_betas, 
                        weight_decay=weight_decay,
                        with_cuda=with_cuda, 
                        cuda_devices=cuda_devices, 
                        log_freq=log_freq)

# %%
print("Training Start")
for epoch in range(epochs):
    svn_model_trained = svn_trainer.train(epoch)
    if epoch%1==0:
        svn_trainer.save(epoch, save_path)

    print("Validation starts")
    if valid_dataloader is not None:
        svn_trainer.test(epoch)

print("Testing starts")
if test_dataloader is not None:
    svn_trainer.test_dataloader = test_dataloader
    svn_trainer.test(1)

# %%



