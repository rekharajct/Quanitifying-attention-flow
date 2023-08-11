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
#attention libraries
from attention_utils.attention_graph_util import *
import networkx as nx
import os
import pandas as pd
from attention_utils.attention_graph_util import *
from attention_utils.inflect_utils import *
from attention_utils.util import inflect

from tqdm import tqdm
from scipy.stats import spearmanr
import math


# %%
from bert.dataset import * #dataset class for BERT pretrainig
from model import * #BERT model
from bert.build_vocab import * #buliding vocabulary
from sv_dataset import * # dataset class for SV classification
from sv_trainer import * #trainer class for classifier
from sv_model import * # Classification model

# %%
# =============================================================================
# #Init
# =============================================================================
train_datapath = "data/train.tsv" #train dataset for train bert
valid_datapath = "data/valid.tsv" #valid dataset for BERT
test_datapath = "data/test.tsv" #test dataset for train bert
sv_classfn_model_path = "classfn_saved_models/sv_trained.ep10"
bert_model_path = "bert_saved_models/bert_trained.ep5"
vocab_path = "wiki.vocab"
results_path = "attn_results_UNK_bert_grads/"

#----------------------BERT hyperparameters------------------------
embed_size = 128 #embedding size of BERT
inner_ff_size = embed_size * 4 #size of the inner FFN in BERT
n_layers = 6 #number of BERT layers
n_heads =  8 #number of attention heads
seq_len = 50 #maximum sequene length
grad_index= -1#index of the verb for which gradient to be calculated, check find_grad is false below
find_grad = False #find gradient w.r.t logits
input_embedding = True #to give input embedding as input instead of token ids
mask_token = "UNK"
num_test_examples =2000
#------------------Classifier hyperparameters------------------
batch_size = 3*32 #number of batch_size
epochs = 101 #number of epochs
num_classes = 2 #number of classes


on_memory = True #Loading on memory: true or false
n_workers = 70 #dataloader worker size

#optimizer arguments 
lr=1e-5
weight_decay = 1e-4
adam_betas = (.9, .999)
dropout = 0 #initially 0.1 changed to 0

#dataloader args
kwargs = {'num_workers':n_workers, 'shuffle':False,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}

#cuda device arguments
with_cuda=True  #use cuda or not
cuda_devices = 5 #CUDA device ids

#print log frequency
log_freq =10
n_layers = 6

#load pretrained bert or not
load_bert_pretrain = False
#------------------------------------------------------------------------------------------------------------------------------
#create vocabulary
vocab_words = create_vocab(train_datapath)


# %%
#create train dataset
train_dataset = BERTDataset(train_datapath, vocab_words, seq_len)
vocab = train_dataset.vocab
vocab_size = len(train_dataset.vocab)


#------------------load BERT model----------------------

bert_model = BERT(n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        inner_ff_size = inner_ff_size,
        n_embeddings = vocab_size,
        input_embedding = input_embedding,
        seq_len = seq_len,
        dropout=dropout, 
        grad_idx=grad_index) #create BERT object

if load_bert_pretrain:
    bert_model_state_dict = torch.load(bert_model_path) #load state dict
    bert_model.load_state_dict(bert_model_state_dict) # load state dict onto the model

#--------------------load classifier--------------------------------
sv_classifier = SVClassifier(bert_model,num_classes)
sv_classifier_state_dict = torch.load(sv_classfn_model_path)
sv_classifier.load_state_dict(sv_classifier_state_dict)

# %%
bert_model.grad_idx

# %%
test_data = pd.read_table(test_datapath)
print(test_data.head())

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
verb_infl, noun_infl = gen_inflect_from_vocab(infl_eng, 'attention_utils/notebooks/wiki.vocab')

# Define the mapping
mapping = {'VBP': 1, 'VBZ': 0}

# %%
for index, example in test_data.iterrows():

    sv_classifier.find_grad = False #set to False if not finding gradients w.r.t logits

    if index > num_test_examples:
        break
    if index % 100==0:
        print(index)
    sentence = example['sentence']

    target = mapping[example['verb_pos']]

    #sent to list
    sent_list = sentence.split()
    #print(sent_list)    

    #tokenize
    tokens = [dataset_obj.vocab[w] if w in dataset_obj.vocab else dataset_obj.UNK for w in sent_list]
    #tokens = [dataset_obj.vocab[w] for w in sent_list]
    

    # add CLS and SEP
    if len(tokens) <= seq_len-2:
        tokens = [dataset_obj.CLS] + tokens + [dataset_obj.SEP]
    else:
        tokens = [dataset_obj.CLS] + tokens[:seq_len-2] + [dataset_obj.SEP]
    
    length = len(tokens)
    
    #pad to seq_len
    padding = [dataset_obj.PAD for _ in range(dataset_obj.seq_len - len(tokens))]
    tokens.extend(padding)
    


    #get the verb positon
    verb_position = example["verb_index"] #verb_index starts at 1 in dataset
    #print("vp", verb_position)
    
    #append verb position
    all_examples_vp.append(verb_position)
    tokens[verb_position] = dataset_obj.UNK

    #print("tk", tokens)
    #tensor of input ids
    input_ids = torch.tensor(tokens).long()

    #get the actual verb
    actual_verb = example['verb']
    
    if actual_verb  not in dataset_obj.vocab:
        continue
     
    #get actual verb index
    actual_verb_index = dataset_obj.vocab[actual_verb]

    #get the inflection of the  actual verb
    inflected_verb = verb_infl[actual_verb]

    if inflected_verb not in dataset_obj.vocab:
        continue

    #get inflected verb index
    inflected_verb_index = dataset_obj.vocab[inflected_verb]    
    
    #store input ids
    all_examples_x.append(input_ids)


    #convert inputs ids to tensor
    sv_inputs = torch.tensor(input_ids).long()

     
    #get classifier outputs: dict
    sv_classifier.eval() #comment if not working
    sv_classifier.bert.grad_idx = actual_verb_index

    classifier_output  = sv_classifier(sv_inputs)    
    
    #classifier logits [batch_size X num_classes]
    logits = classifier_output["logits"]

    
    #softmax probs of classes     
    classifier_probs = classifier_output["softmax_probs"]
   
    #list of attention from each encoder each of shape [batch_size X n_layers X seq_len X seq_len]
    layerwise_attentions = classifier_output["attentions"]


    #store the attention matrices
    _attentions = [att.detach().numpy() for att in layerwise_attentions]
    attentions_mat = np.asarray(_attentions)[:,0] #shape [n_layers X n_heads X seq_len X seq_len]
    all_examples_attentions.append(attentions_mat)  

    #output of encoders: list of encoder ouputs of shape [batch_size X seq_len X embed_size]
    bert_hidden_states = classifier_output["bert_hidden_states"]    
    
    #bert embedding: [batch_size X seq_len X embed_size]
    bert_embedding = classifier_output["bert_embedding"]

    #gradients of softmax probs of  actual verb of bert prediction  w.r.t bert embedding  [batch_size X seq_len X embed_size]
    bert_embed_grads = classifier_output["bert_embed_grads"]  
     
    #get bert prediction logits [batch_size X seq_len X vocab_size]
    bert_pred_logits = classifier_output["bert_predict_logits"]
    
    #get bert prediction probabilities
    probs =  torch.squeeze (nn.functional.softmax(bert_pred_logits, dim=-1))

    #bert pred probability for actual verb
    actual_verb_score = probs[verb_position][actual_verb_index]
    
    #prob of inflectd verb score by BERT
    inflected_verb_score = probs[verb_position][inflected_verb_index]

    #diff btw actual verb score and inflected verb score
    main_diff_score = actual_verb_score - inflected_verb_score

    #store the main diff  score
    all_examples_accuracies.append(main_diff_score > 0)

    
    #get grad of classifier logits w.r.t CLS token embedding
    classifier_grad_dict = classifier_output["classifier_grad"]

    if find_grad: #if find grad w.r.t classifier logits

        #get the gradient for the target [batch_size X seq_len X embed_size] , here batch_size is 1
        grad_target = classifier_grad_dict[target]
        
        #get grad scores 
        grad_target_array = grad_target.detach().numpy() # [seq_len X embed_size] 
        #print("gta",grad_target_array.shape)
        grad_scores = abs(np.sum(grad_target_array, axis=-1))
        #print("gs",grad_scores.shape)

        #grad_scores = np.reshape(grad_scores, (1,-1))
    
    else:
        grad_scores = 0
        grad_scores = abs(np.sum(bert_embed_grads.detach().numpy(), axis=-1))
    #store grad_scores
    all_examples_grads.append(grad_scores)

    #reset classifier gradient computation  to False
    sv_classifier.find_grad = False

    #reset the vocab index for which gradient is to be calculated
    sv_classifier.bert.grad_idx = -1


    # Repeating examples and replacing one token at a time with unk
    batch_size = 1
    max_len = input_ids.shape[0]

    # Repeat each example 'max_len' times
    x = torch.unsqueeze(input_ids, dim=0) # reshape to [1 X size]
    # extend x [max_len X max_len]
    extended_x = np.reshape(np.tile(x[:,None,...], (1, max_len, 1)),(-1,x.shape[-1]))
    

    # Create unk sequences and unk mask
    if mask_token =="UNK":
        unktoken = dataset_obj.UNK
    else:
        unktoken = dataset_obj.MASK

    #print(unktoken)
    unks = unktoken * np.eye(max_len)
    unks =  np.tile(unks, (batch_size, 1)) #[max_len X max_len]
    
    unk_mask =  (unktoken - unks)/unktoken

    # Replace one token in each repeatition with unk
    extended_x = extended_x * unk_mask + unks

    #convert extended_x to tensor
    ext_x = torch.tensor(extended_x).long()


    #get the new classifier output
    extended_out = sv_classifier(ext_x)

    #get extended prediction logits  [seq_len X seq_len X vocab_size]
    extended_logits = extended_out["bert_predict_logits"] 


    #get softmax probs
    extended_probs = nn.functional.softmax(extended_logits, dim=-1)

    #get extended correct probs
    extended_correct_probs = extended_probs[:,verb_position,actual_verb_index]
    

    extended_wrong_probs =  extended_probs[:,verb_position,inflected_verb_index]
    extended_diff_scores = extended_correct_probs - extended_wrong_probs

    # Save the difference in the probability predicted for the correct class
    diffs = abs(main_diff_score - extended_diff_scores)


    all_examples_blankout_relevance.append(diffs.detach())
    
    num_test_examples = num_test_examples -1
    

# %%
all_examples_vp

# %%
print("Writing results to files")
import pickle
with open(results_path +'all_examples_x', 'wb') as fp:
    pickle.dump(all_examples_x, fp)


with open(results_path +'all_examples_vp', 'wb') as fp:
    pickle.dump(all_examples_vp, fp)

with open(results_path +'all_examples_y', 'wb') as fp:
    pickle.dump(all_examples_y, fp)


with open(results_path +'all_examples_attentions', 'wb') as fp:
    pickle.dump(all_examples_attentions, fp)

with open(results_path +'all_examples_blankout_relevance', 'wb') as fp:
    pickle.dump(all_examples_blankout_relevance, fp)

with open(results_path +'all_examples_grads', 'wb') as fp:
    pickle.dump(all_examples_grads, fp)

"""
with open(results_path +'all_examples_inputgrads', 'wb') as fp:
    pickle.dump(all_examples_inputgrads, fp)
"""

with open(results_path +'all_examples_accuracies', 'wb') as fp:
    pickle.dump(all_examples_accuracies, fp)

print("Completed")
