import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import warnings

class MicrobeTransformer(nn.Module):
    def __init__(self, embedding_dim, n_heads, padding=True):
        super(MicrobeTransformer, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        if padding:
            self.embedding = nn.Embedding(73, self.embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(73, self.embedding_dim)
        self.position_embeddings = PositionalEncoding(self.embedding_dim, max_len=37, dropout_rate=0.1)
        self.transformer = MultiHeadAttention(d_model_1=self.embedding_dim, d_model_2=self.embedding_dim,
                                              n_heads=self.n_heads, d_model=self.embedding_dim)

    def forward(self, v):
        v = self.embedding(v.long())
        v = self.position_embeddings(v.transpose(0,1)).transpose(0,1)
        v = self.transformer(v)
        return v

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Embeddings(nn.Module):
    #Vanilla embedding + positional embedding :)
    def __init__(self, vocab_size, hidden_size, max_len):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEncoding(hidden_size, max_len, dropout_rate=0.1)
    
    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)    
        embeddings = self.position_embeddings(words_embeddings.transpose(0,1)).transpose(0,1)
        return embeddings


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, len_q, len_k] 
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #[batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Automatically brocast to n_head on dimension 1
        attn = nn.Softmax(dim=-1)(scores) #calculate total attntion value of a single position
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, n_heads,d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q_dense = nn.Linear(self.d_model_1, self.d_model * self.n_heads, bias=False) 
        self.W_K_dense = nn.Linear(self.d_model_2, self.d_model * self.n_heads, bias=False)
        self.W_V_dense = nn.Linear(self.d_model_2, self.d_model * self.n_heads, bias=False)
        
        self.scale_product = ScaledDotProductAttention(self.d_model)
        self.out_dense = nn.Linear(self.n_heads * self.d_model, self.d_model, bias=False)  # self.n_heads * self.d_dim = const
        self.LN = nn.LayerNorm(self.d_model)
        
    def forward(self, X, attn_mask=None):
        Q_residual, batch_size = X, X.size(0)
        # (B, S[seqlen], D[d_model]) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q_dense(X).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        k_s = self.W_K_dense(X).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        self.v_s = self.W_V_dense(X).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_q, seq_k]
        context = self.scale_product(q_s, k_s, self.v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        context = self.out_dense(context)
        output = context + Q_residual
        output = self.LN(output)
        return output


