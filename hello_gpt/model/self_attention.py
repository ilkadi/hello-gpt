import torch
import torch.nn as nn
from hello_gpt.model.abc_module import AbstractGPTModule

""" Meowalization diagram for the Self Attention module:
+----------------+      +------------------+      +------------------+      +-------------------+      +-----------------+
| Input Tensor   |      | Combined         |      | Reshape &        |      | Scaled Dot        |      | Combined        |
| (batch_size,   | ---> | Attention        | ---> | Transpose for    | ---> | Product           | ---> | Projection      |
| seq_length,    |      | (Linear Layer)   |      | Multi-Head       |      | Attention         |      | (Linear Layer)  |
| n_embd)        |      | n_embd to        |      | Attention        |      | (n_head, seq_len, |      | n_embd to       |
|                |      | 3*n_embd         |      |                  |      | seq_len)          |      | n_embd          |
+----------------+      +------------------+      +------------------+      +-------------------+      +-----------------+
                                 |                                                    ^
                                 +---------------------+------------------------------+
                                                       |
                                                   +------------------+
                                                   |                  |
                                                   | Residual Dropout |
                                                   | (config.dropout) |
                                                   |                  |
                                                   +------------------+
"""
class SelfAttention(AbstractGPTModule):
    def __init__(self, n_embd, n_head, dropout, bias=False):
        super().__init__()
        # Names are essential for GPT2 model loading
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout   = nn.Dropout(dropout)
        self.resid_dropout  = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size() 
        queries, keys, values  = self.c_attn(x).split(self.n_embd, dim=2)
        
        keys = keys.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)
        queries = queries.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)

        # causal self-attention: (batch_size, n_head, seq_length, s_head) x (batch_size, n_head, s_head, seq_length) -> (batch_size, n_head, seq_length, seq_length)
        dropout_p = self.dropout if self.training else 0
        y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y