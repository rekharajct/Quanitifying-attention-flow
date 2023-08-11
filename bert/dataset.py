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
class BERTDataset(Dataset):
    #Init dataset
    def __init__(self, datapath, vocab, seq_len):
        dataset = self
        dataset.data_df = pd.read_table(datapath)
        dataset.texts = dataset.data_df["sentence"]
        dataset.sentences = [s.split() for s in dataset.texts]
        # Define the mapping
        mapping = {'VBP': 1, 'VBZ': 0}

        # Convert labels to numbers using the mapping
        dataset.sv_targets = dataset.data_df["verb_pos"].map(mapping).tolist()
    
        dataset.vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + vocab
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.rvocab = {v:k for k,v in dataset.vocab.items()}
        dataset.seq_len = seq_len
        
        #special tags
        dataset.PAD = dataset.vocab['[PAD]'] #replacement tag for tokens to ignore
        dataset.UNK = dataset.vocab['[UNK]'] #replacement tag for unknown words
        dataset.MASK = dataset.vocab['[MASK]'] #replacement tag for the masked word prediction task
        dataset.CLS= dataset.vocab['[CLS]'] #Classification token
        dataset.SEP = dataset.vocab['[SEP]' ] # EOS token

    
    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self       
        sentence  = dataset.sentences[index]
        

        #random mask
        sent_masked, sent_unmasked =   dataset.apply_random_mask(sentence, p_random_mask)  
        sent_masked = [dataset.CLS] + sent_masked + [dataset.SEP]
        sent_unmasked = [dataset.PAD]   + sent_unmasked + [dataset.PAD]

        #ensure that the sequence is of length seq_len
        bert_input = sent_masked[:dataset.seq_len]
        bert_target = sent_unmasked[:dataset.seq_len]
        
        padding = [dataset.PAD for _ in range(dataset.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_target.extend(padding)

        return {'input': torch.Tensor(bert_input).long(),
                'target': torch.Tensor(bert_target).long(),
                'sv_target': torch.Tensor(dataset.sv_targets)}
        

    #return length
    def __len__(self):
        return len(self.sentences)
    
    def apply_random_mask(self, tokens, random_prob):
        dataset = self
        #tokens = sentence.split()
        token_idx = [0] * len(tokens)
        output_label = []
        
        for i, token in enumerate(tokens):
            prob = random.random()
            
            if prob < random_prob:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    
                    token_idx[i] = dataset.MASK

                # 10% randomly change token to random token
                elif prob < 0.9:
                
                    token_idx[i] = random.randrange(len(dataset.vocab)-1)
                
                # 10% randomly change token to current token
                else:
                    if token in dataset.vocab:
                        token_idx[i] = dataset.vocab[token]
                    else:
                        token_idx[i] = dataset.UNK
                        
                if token in dataset.vocab:
                    output_label.append(dataset.vocab[token])
                else:
                    output_label.append(dataset.UNK)
            else:
                if token in dataset.vocab:
                    token_idx[i] = dataset.vocab[token]
                else:
                    token_idx[i] = dataset.UNK
                output_label.append(0)
        return token_idx, output_label


    #get words id
    def get_sentence_idx(self, sent):
        dataset = self
        sent_tokenised = [dataset.vocab[w] if w in dataset.vocab else dataset.UNK for w in sent] 
        return sent_tokenised
    




    
    
    
