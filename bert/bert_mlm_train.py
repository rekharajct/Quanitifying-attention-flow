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
from dataset import *
from model import *
from build_vocab import *
from trainer import *
from bert_predict import *

# %%


# %% [markdown]
# Hyperparameters

# %%
# =============================================================================
# #Init
# =============================================================================
train_datapath = "data/train.tsv" #train dataset for train bert
valid_datapath = "data/valid.tsv" #valid dataset for BERT
test_datapath = "data/test.tsv" #test dataset for train bert
vocab_path = "wiki.vocab"
save_path = "saved_models/bert_trained"
embed_size = 128 #embedding size of BERT
inner_ff_size = embed_size * 4 #size of the inner FFN in BERT
n_layers = 6 #number of BERT layers
n_heads =  8 #number of attention heads
seq_len = 50 #maximum sequene length
batch_size = 32*3 #number of batch_size
epochs = 31 #number of epochs

on_memory = True #Loading on memory: true or false
n_workers = 80 #dataloader worker size

#optimizer arguments 
#lr=2e-3
lr = 0.001
weight_decay = 1e-4
adam_betas = (.9, .999)
dropout = 0.1

#dataloader args
kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}

#cuda device arguments
with_cuda= True #use cuda or not
cuda_devices = [4,5,6] #CUDA device ids

#print log frequency
log_freq =10

#to compute gradient or not of vocab_index
grad_index = 1

# %% [markdown]
# Create vocabulary

# %%
vocab = create_vocab_wiki(vocab_path)

# %% [markdown]
# create train, valid and test datasets

# %%
#create train dataset
train_dataset = BERTDataset(train_datapath, vocab, seq_len)
valid_dataset = BERTDataset(valid_datapath, vocab, seq_len)
test_dataset = BERTDataset(test_datapath, vocab, seq_len)
vocab_size = len(train_dataset.vocab)
print(vocab_size)

# %% [markdown]
# create dataloaders

# %%
train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **kwargs)
test_dataloader = torch.utils.data.DataLoader(test_dataset, **kwargs)

# %% [markdown]
# Instantiate BERT model
# 

# %%
# =============================================================================
# Model
# =============================================================================
#build BERT model
bert_model = BERT(n_layers, n_heads, embed_size, inner_ff_size,\
              vocab_size, seq_len, dropout, grad_index)



# %% [markdown]
# Create BERT trainer

# %%
#Creating BERT Trainer
bert_trainer = BERTTrainer(bert_model,
                        len(vocab), 
                        train_dataloader=train_dataloader, 
                        test_dataloader=valid_dataloader,
                        lr=lr, 
                        betas=adam_betas, 
                        weight_decay=weight_decay,
                        with_cuda=with_cuda, 
                        cuda_devices=cuda_devices, 
                        log_freq=log_freq)

# %% [markdown]
# Start Training

# %%
print("Training Start")
for epoch in range(epochs):
    bert_model_trained = bert_trainer.train(epoch)
    if epoch %1 ==0:
        bert_trainer.save(epoch, save_path)

    print("Vaidation starts")
    if valid_dataloader is not None:
        bert_trainer.test(epoch)

print("Testing starts")
if test_dataloader is not None:
    bert_trainer.test_dataloader = test_dataloader
    bert_trainer.test(1)
