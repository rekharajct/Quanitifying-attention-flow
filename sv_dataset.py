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
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Dataset
# =============================================================================
class SVDataset(Dataset):
    #Init dataset
    def __init__(self, datapath, vocab, seq_len):
        dataset = self
        dataset.data_df = pd.read_table(datapath)
        dataset.texts = dataset.data_df["sentence"]
        dataset.sentences = [s.split() for s in dataset.texts]
        dataset.verbs = dataset.data_df["verb"]

        #get the verb_positon
        dataset.verb_pos = dataset.data_df["verb_index"]
        # Define the mapping
        mapping = {'VBP': 1, 'VBZ': 0}

        # Convert labels to numbers using the mapping
        dataset.sv_targets = dataset.data_df["verb_pos"].map(mapping).tolist()
    
        dataset.vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + vocab
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.rvocab = {v:k for k,v in dataset.vocab.items()}
        dataset.seq_len = seq_len
        
        #special tags
        dataset.PAD = dataset.vocab['[PAD]'] #replacement tag for tokens to ignore  - 0
        dataset.UNK = dataset.vocab['[UNK]'] #replacement tag for unknown words 4
        dataset.MASK = dataset.vocab['[MASK]'] #replacement tag for the masked word prediction task - 3
        dataset.CLS= dataset.vocab['[CLS]'] #Classification token -1
        dataset.SEP = dataset.vocab['[SEP]' ] # EOS token -2

    
    #fetch data
    def __getitem__(self, index):
        dataset = self       
        sentence  = dataset.sentences[index]
        target = dataset.sv_targets[index]
        text = dataset.texts[index]
        verb = dataset.verbs[index]
        sentence = dataset.sentences[index]

        #get verb position
        verb_position = dataset.verb_pos[index] # in the dataset pos starts from 1 

               
        #tokenize the sentence
        sent_tokenized = dataset.tokenize(sentence)
        sent_tokenized = [dataset.CLS] + sent_tokenized + [dataset.SEP]

        #replace verb with UNK
        sent_tokenized[verb_position] = dataset.UNK


        #ensure that the sequence is of length seq_len
        sv_input = sent_tokenized[:dataset.seq_len]
        #print(sv_input)

        #apply padding
        padding = [dataset.PAD for _ in range(dataset.seq_len - len(sv_input))]
        sv_input.extend(padding)
        


        return {'input': torch.Tensor(sv_input).long(),
                'sv_target': torch.Tensor([target]).long()}
        

    #return length
    def __len__(self):
        return len(self.sentences)
    
    def tokenize(self, tokens):
        """
        Tokenize the sentence.
        param tokens: list of words in a sentence
        return token ids of the words as list
        """
        dataset = self
        token_idx = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in dataset.vocab:
                token_idx[i]=dataset.vocab[token]
            else:
                token_idx[i]=dataset.UNK
        return token_idx



    
