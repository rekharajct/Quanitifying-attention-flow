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

def create_vocab_wiki(vocabpath):
    with open(vocabpath, 'r') as file:
        lines = file.readlines()

    #3) create vocab if not already created
    print('creating/loading vocab...')

    words = []
    #process each line and extract words
    for line  in lines:
        parts = line.split()
        if len(parts)>=1:
            word = parts[0]  
            words.append(word)
    #print(words)
    vocab_counter = Counter(words) #keep the N most frequent words
    #print(vocab_counter)
    vocab = [w for w in vocab_counter]
    return vocab
    
    
def create_vocab(datapath):
    data_set = pd.read_table(datapath)
    sentences = data_set["sentence"]
    #2) tokenize sentences (can be done during training, you can also use spacy udpipe)
    print('tokenizing sentences...')
    sentence_list = [s.split() for s in sentences]
    #print(sentence_list)

    #3) create vocab if not already created
    print('creating/loading vocab...')
    vocab_path = 'vocab.txt'
    words = []
    for sent in sentence_list:
            #print(sent)
            for w in sent:
                #print("w", w)
                words.append(w)
    #print(words)
    vocab_counter = Counter(words) #keep the N most frequent words
    #print(vocab_counter)
    vocab = [w for w in vocab_counter]
    return vocab

