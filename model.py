
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



# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    """Compute scaled dot product attentions
        :param q, k, v
        :return attention and context
    """
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output, scores

class MultiHeadAttention(nn.Module):
    """Compute MHA"""
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
#        self.q_linear = nn.Linear(out_dim, out_dim)
#        self.k_linear = nn.Linear(out_dim, out_dim)
#        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        
        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
        
        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        #n_heads => attention => merge the heads => mix information
        scores, attn = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
        
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #inp => inner => gelu => dropout => inner => inp
        x = self.linear1(x)
        x = F.gelu(x, approximate='tanh') # Using tanh approximation as it is more accurate
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x2, attn = self.mha(x2, mask=mask)
        x2_drop1 = self.dropout1(x2)
        x = x + x2_drop1
        x2 = self.norm2(x)
        x2 = self.ff(x2)
        x2_drop2 = self.dropout2(x2)
        x = x + x2_drop2
        return x, attn

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len
    

def gradients_hook(grad_dict, layer_name, grad_input, grad_output): 
    """function to store gradients
    param grad_dict: dict to store layerwise gradient
    layer_name: name given to layer
    grad_input: gradient of loss w.r.t input of  a module
    grad_output: gradient of loss w.r.t output of a module
    """
    grad_dict[layer_name] = {}
    grad_dict[layer_name]["grad_input"] = grad_input
    grad_dict[layer_name]["grad_output"] = grad_output

class BERT(nn.Module):
    """
        :param n_embeddings: vocab_size of total words
        :param embed_size: BERT model embedding size
        :param n_layers: numbers of Transformer blocks(layers)
        :param n_heads: number of attention heads
        :param dropout: dropout rate
        :param inner_ff_size: inner FFN size
        """
    def __init__(self, n_layers, n_heads, embed_size, inner_ff_size, n_embeddings,input_embedding, seq_len, dropout=.1, grad_idx=None):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_size = embed_size
        self.inner_ff_size = inner_ff_size
        self.n_embeddings = n_embeddings
        self.seq_len = seq_len
        self.grad_dict = {}
        self.grad_idx = grad_idx
        self.input_embedding =  input_embedding

        
        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)

        #register backward hook on input embedding layer
        self.embed_handle = self.embeddings.register_full_backward_hook(self.embed_backward_hook)
        self.pe = PositionalEmbedding(embed_size, seq_len)
        
        
        #encoder layers
        encoders = []
        for i in range(n_layers):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)
        
        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
        self.softmax = nn.Softmax(dim=-1)
                
    
    def forward(self, x):

        if self.input_embedding:
            x_emb  = x
        else:
            x_emb = self.embeddings(x)
     
        x = x_emb + self.pe(x_emb)

        layerwise_attn = [] #to store attention at every layer
        layerwise_hidden_states = [] #to store hidden states of each layer
        for encoder in self.encoders:
            x, attn = encoder(x)
            layerwise_attn.append(attn)
            layerwise_hidden_states.append(x)

        bert_embedding = self.norm(x)
        
        out = self.linear(bert_embedding)

        probs = self.softmax(out)
    

        emb_grads = None # to store the embedding gradients w.r.t y of a particular index
        

        if self.grad_idx>=0:

            #gradients of the bert embdding w.r.t to softmax prob
            emb_grads = torch.autograd.grad(outputs=probs[:, :, self.grad_idx], inputs=x_emb, \
                                            grad_outputs =torch.ones_like(probs[:, :, self.grad_idx]),  retain_graph=True)[0]
        
        
        return {"input_embedding":x_emb,"inp_grad":self.grad_dict, "bert_embedding":bert_embedding, "bert_prediction": out,
                "attention":layerwise_attn, 
                "hidden_states":layerwise_hidden_states, "bert_emb_grads": emb_grads }

    def embed_backward_hook(self, module, grad_input, grad_output):  
        """
        backward hook function
        """
        gradients_hook(self.grad_dict, "embed_layer", grad_input, grad_output)
