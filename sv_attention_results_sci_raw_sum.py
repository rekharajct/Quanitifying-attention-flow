# %%
# =============================================================================
# Libs
# =============================================================================
import torchvision
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import math
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#attention libraries
from attention_utils.attention_graph_util import *
import networkx as nx
import os
import pandas as pd
from attention_utils.attention_graph_util import *
#from attention_utils.notebooks.notebook_utils import *
from attention_utils.inflect_utils import *
from attention_utils.util import inflect

from tqdm import tqdm
import math
import pandas as pd
from scipy import stats

# %%
from bert.dataset import * #dataset class for BERT pretrainig
from model import * #BERT model
from build_vocab import * #buliding vocabulary
from sv_dataset import * # dataset class for SV classification
from sv_trainer import * #trainer class for classifier
from sv_model import * # Classification model

# %%
# =============================================================================
# #Init
# =============================================================================
#dirpath = "/lfs/usrhome/phd/cs22d010/research/bert_trained1/"
dirpath = ""
train_datapath =dirpath+ "data/train.tsv" #train dataset for train bert
valid_datapath = dirpath+"data/valid.tsv" #valid dataset for BERT
test_datapath = dirpath+"data/test.tsv" #test dataset for train bert


vocab_path = dirpath+"wiki.vocab"
#results_path = dirpath+"attn_results_UNK_class_grads/"
#results_path = dirpath + "attn_results_MASK_class_grads/"
results_path = dirpath + "attn_results_UNK_bert_grads/"
#results_path = dirpath + "attn_results_MASK_class_grads/"


embed_size = 128 #embedding size of BERT
seq_len = 50 #maximum sequene length
batch_size = 3*32 #number of batch_size
epochs = 50 #number of epochs
num_classes = 2 #number of classes

on_memory = True #Loading on memory: true or false
n_workers = 80 #dataloader worker size

#optimizer arguments 
lr=1e-5
weight_decay = 1e-4
adam_betas = (.9, .999)
dropout = 0 #initially 0.1 changed to 0

#dataloader args
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}

#cuda device arguments
with_cuda= True #use cuda or not
cuda_devices = None #CUDA device ids

#print log frequency
log_freq =10
n_layers = 6


min_id=0
max_id=1999

find_flow = False

#-----------------------------------------------------------------------------------------------------------------
# _____________________________________________________________________________________________________

#create vocabulary
vocab_words = create_vocab(train_datapath)


# %%
#create train dataset
train_dataset = BERTDataset(train_datapath, vocab_words, seq_len)
vocab = train_dataset.vocab

# %%
test_data = pd.read_table(test_datapath)

# %%
dataset_obj = train_dataset


# %%
all_examples_x = [] #store all examples
all_examples_vp = [] # to store verb position
all_examples_y = []

all_examples_attentions = []
all_examples_blankout_relevance = []
all_examples_grads = [] #store grad scores
all_examples_inputgrads = [] #store input grad scores
n_batches = 1000

all_examples_accuracies = [] #store main diff score

infl_eng = inflect.engine()


# %%
verb_infl, noun_infl = gen_inflect_from_vocab(infl_eng, vocab_path)

# %%
import pickle
print("files opened")
with open(results_path +'all_examples_x', 'rb') as fp:
    all_examples_x=pickle.load(fp)

with open(results_path +'all_examples_vp', 'rb') as fp:
    all_examples_vp=pickle.load(fp)

with open(results_path +'all_examples_y', 'rb') as fp:
    all_examples_y=pickle.load(fp)


with open(results_path +'all_examples_attentions', 'rb') as fp:
    all_examples_attentions=pickle.load(fp)

with open(results_path +'all_examples_blankout_relevance', 'rb') as fp:
    all_examples_blankout_relevance=pickle.load(fp)

with open(results_path +'all_examples_grads', 'rb') as fp:
    all_examples_grads=pickle.load(fp)

"""
with open(results_path +'all_examples_inputgrads', 'rb') as fp:
    all_examples_inputgrads=pickle.load(fp)
"""

with open(results_path +'all_examples_accuracies', 'rb') as fp:
    all_examples_accuracies=pickle.load(fp)

print("lists created")


all_examples_x = all_examples_x[min_id:max_id] #store all examples
all_examples_vp =all_examples_vp[min_id:max_id] # to store verb position
all_examples_y =all_examples_y[min_id:max_id]

all_examples_attentions = all_examples_attentions[min_id:max_id]
all_examples_blankout_relevance = all_examples_blankout_relevance[min_id:max_id]
all_examples_grads =all_examples_grads[min_id:max_id] #store grad scores
#all_examples_inputgrads = all_examples_inputgrads[min_id:max_id]


def spearmanr1(x, y):
    """ `x`, `y` --> pd.Series"""
    x = pd.Series(x)
    y = pd.Series(y)
    assert x.shape == y.shape
    rx = x.rank(method='dense')
    ry = y.rank(method='dense')
    d = rx - ry
    dsq = np.sum(np.square(d))
    n = x.shape[0]
    coef = 1. - (6. * dsq) / (n * (n**2 - 1.))
    return [coef]

def get_raw_att_relevance_max(full_att_mat, input_tokens, layer=-1, output_index=0):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    return att_sum_heads[layer].max(axis=0)

def get_raw_att_relevance(full_att_mat, input_tokens, layer=-1, output_index=0):
    raw_rel = full_att_mat[layer].sum(axis=0)[output_index]/full_att_mat[layer].sum(axis=0)[output_index].sum()
            
    return raw_rel


def get_flow_relevance(full_att_mat, input_tokens, layer, output_index):
        
        input_tokens = input_tokens
        res_att_mat = full_att_mat.sum(axis=1)/full_att_mat.shape[1]
        res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
        res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

        res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_tokens)
                            
        A = res_adj_mat
        res_G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')


        output_nodes = ['L'+str(layer+1)+'_'+str(output_index)]
        input_nodes = []
        for key in res_labels_to_index:
            if res_labels_to_index[key] < full_att_mat.shape[-1]:
                input_nodes.append(key)
                                                                                      
        flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes=input_nodes, output_nodes=output_nodes, length=full_att_mat.shape[-1])
                                                                                                                        
        n_layers = full_att_mat.shape[0]
        length = full_att_mat.shape[-1]
        final_layer_attention = flow_values[(layer+1)*length:, layer*length:(layer+1)*length]
        relevance_attention_flow = final_layer_attention[output_index]

        return relevance_attention_flow
    
    

def get_joint_relevance(full_att_mat, input_tokens, layer=-1, output_index=0):
        att_sum_heads =  full_att_mat.sum(axis=1) / full_att_mat.shape[1]
        joint_attentions = compute_joint_attention(att_sum_heads, add_residual=True)
        relevance_attentions = joint_attentions[layer][output_index]
        return relevance_attentions

# %%
print("compute raw relevance scores ...")
all_examples_raw_relevance = {}
for l in np.arange(0,n_layers):
    all_examples_raw_relevance[l] = []
    for i in np.arange(len(all_examples_x)):
        #get the tokens
        enc_example_tensor = all_examples_x[i]
        enc_example = enc_example_tensor.tolist()
        tokens = [dataset_obj.rvocab[w] for w in enc_example ]
        #get verb position
        vp = all_examples_vp[i]
        length = len(tokens)
        attention_relevance = get_raw_att_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
        all_examples_raw_relevance[l].append(np.asarray(attention_relevance))
    

# %%
print("compute joint relevance scores ...")
all_examples_joint_relevance = {}
for l in np.arange(0,n_layers):
    all_examples_joint_relevance[l] = []
    for i in np.arange(len(all_examples_x)):
        #get the tokens
        enc_example_tensor = all_examples_x[i]
        enc_example = enc_example_tensor.tolist()
        tokens = [dataset_obj.rvocab[w] for w in enc_example ]
        
        vp = all_examples_vp[i]
        length = len(tokens)
        attention_relevance = get_joint_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
        all_examples_joint_relevance[l].append(np.asarray(attention_relevance))
        

# %%
import time
print("compute flow relevance scores ...")
all_examples_flow_relevance = {}
if find_flow:
    for l in np.arange(0,6):
        
        all_examples_flow_relevance[l] = []
        for i in np.arange(len(all_examples_x)):
            start_time = time.time()
            #get the tokens
            enc_example_tensor = all_examples_x[i]
            enc_example = enc_example_tensor.tolist()
            tokens = [dataset_obj.rvocab[w] for w in enc_example ]
            #get verb position
            vp = all_examples_vp[i]
            length = len(tokens)
            attention_relevance = get_flow_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
            all_examples_flow_relevance[l].append(np.asarray(attention_relevance))
            end_time = time.time()
            print("time in sec for layer" + str(l) +" pt " + str(i) + ":", end_time - start_time)

# %%
raw_sps_blank = []
raw_sps_grad = []
raw_sps_inputgrad = []

joint_sps_blank = []
joint_sps_grad = []
joint_sps_inputgrad = []

flow_sps_blank = []
flow_sps_grad = []
flow_sps_inputgrad = []

if find_flow:
    cols =  ["raw blankout mean", "raw blankout std", "raw grad mean", "raw grad std", \
        "joint blankout mean", "joint blankout std", "joint grad mean", "joint grad std",\
        "flow mean", "flow std", "flow grad mean", "flow grad std" ]
else:
    cols =  ["raw blankout mean", "raw blankout std", "raw grad mean", "raw grad std", \
        "joint blankout mean", "joint blankout std", "joint grad mean", "joint grad std",\
        ]

data={col:[] for col in cols}


# %%
for l in np.arange(0,n_layers):
    print("###############Layer ",l, "#############")
    print('raw blankout')
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_raw_relevance[l][i],all_examples_blankout_relevance[i].numpy())
        spc = sp.correlation
        if not math.isnan(spc):
            raw_sps_blank.append(spc)
        else:
            raw_sps_blank.append(0)
        
    #print(np.mean(raw_sps_blank), np.std(raw_sps_blank))
    data["raw blankout mean"].append(np.mean(raw_sps_blank))
    data["raw blankout std"].append(np.std(raw_sps_blank))
    
    """"
    print('raw inputgrad')
    print(all_examples_raw_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_raw_relevance[l][i],all_examples_inputgrads[i][0])
        spc = sp.correlation
        if not math.isnan(spc):
            raw_sps_inputgrad.append(spc)
        else:
            raw_sps_inputgrad.append(0)
        
    #print(np.mean(raw_sps_inputgrad), np.std(raw_sps_inputgrad))
    data["raw inputgrad mean"].append(np.mean(raw_sps_inputgrad))
    data["raw inputgrad std"].append(np.std(raw_sps_inputgrad))
    """
    print('raw grad')
    for i in np.arange(len(all_examples_x)):
        print("raw",all_examples_raw_relevance[l][i].shape, "grad", all_examples_grads[i].shape)
        sp = stats.spearmanr(all_examples_raw_relevance[l][i],all_examples_grads[i])
        spc = sp.correlation
        if not math.isnan(spc):
            raw_sps_grad.append(spc)
        else:
            raw_sps_grad.append(0)
        
    #print(np.mean(raw_sps_grad), np.std(raw_sps_grad))
    data["raw grad mean"].append(np.mean(raw_sps_grad))
    data["raw grad std"].append(np.std(raw_sps_grad))

    print('joint blankout')
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_joint_relevance[l][i],all_examples_blankout_relevance[i].numpy())
        spc = sp.correlation
        if not math.isnan(spc):
            joint_sps_blank.append(spc)
        else:
            joint_sps_blank.append(0)
        
    #print(np.mean(joint_sps_blank), np.std(joint_sps_blank))
    data["joint blankout mean"].append(np.mean(joint_sps_blank))
    data["joint blankout std"].append(np.std(joint_sps_blank))

    print('joint grad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_grads[i][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_joint_relevance[l][i],all_examples_grads[i])
        spc = sp.correlation
        if not math.isnan(spc):
            joint_sps_grad.append(spc)
        else:
            joint_sps_grad.append(0)
        
    #print(np.mean(joint_sps_grad), np.std(joint_sps_grad))
    data["joint grad mean"].append(np.mean(joint_sps_grad))
    data["joint grad std"].append(np.std(joint_sps_grad))    

    """
    print('joint inputgrad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_joint_relevance[l][i],all_examples_inputgrads[i][0])
        spc = sp.correlation
        if not math.isnan(spc):
            joint_sps_inputgrad.append(spc)
        else:
            joint_sps_inputgrad.append(0)
        
    #print(np.mean(joint_sps_inputgrad), np.std(joint_sps_inputgrad))
    data["joint inputgrad mean"].append(np.mean(joint_sps_inputgrad))
    data["joint inputgrad std"].append(np.std(joint_sps_inputgrad))
    """
    if find_flow:
        print('flow')
        for i in np.arange(len(all_examples_x)):
            sp = stats.spearmanr(all_examples_flow_relevance[l][i],all_examples_blankout_relevance[i].numpy())
            spc = sp.correlation
            if not math.isnan(spc):
                flow_sps_blank.append(spc)
            else:
                flow_sps_blank.append(0)
            
        #print(np.mean(flow_sps_blank), np.std(flow_sps_blank))
        data["flow mean"].append(np.mean(flow_sps_blank))
        data["flow std"].append(np.std(flow_sps_blank))


        print('flow grad')
        print(all_examples_joint_relevance[l][0].shape, all_examples_grads[0][0].shape)
        for i in np.arange(len(all_examples_x)):
            sp = stats.spearmanr(all_examples_flow_relevance[l][i],all_examples_grads[i])
            spc = sp.correlation
            if not math.isnan(spc):
                flow_sps_grad.append(spc)
            else:
                flow_sps_grad.append(0)
            
        #print(np.mean(flow_sps_grad), np.std(flow_sps_grad))
        data["flow grad mean"].append(np.mean(flow_sps_grad))
        data["flow grad std"].append(np.std(flow_sps_blank))

    """
    print('flow inputgrad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = stats.spearmanr(all_examples_flow_relevance[l][i],all_examples_inputgrads[i][0])
        spc = sp.correlation
        if not math.isnan(spc):
            flow_sps_inputgrad.append(spc)
        else:
            flow_sps_inputgrad.append(0)
        
    #print(np.mean(flow_sps_inputgrad), np.std(flow_sps_inputgrad))
    data["flow inputgrad mean"].append(np.mean(flow_sps_inputgrad))
    data["flow inputgrad std"].append(np.mean(flow_sps_inputgrad))
    """
results_df = pd.DataFrame(data)
#results_df.to_csv(dirpath + "results_UNK_class_grads.csv")
#results_df.to_csv(dirpath + "results_MASK_class_grads.csv")
results_df.to_csv(dirpath + "results_UNK_bert_grads.csv")
#results_df.to_csv(dirpath + "results_MASK_class_grads.csv")





