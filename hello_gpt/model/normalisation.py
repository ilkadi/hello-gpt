import torch
import torch.nn as nn
from torch.nn import functional as F
from hello_gpt.model.abc_module import AbstractGPTModule

""" Meowalization diagram for the Normalization module:
+---------------+         +-------------------+         +---------------+
| Input Tensor  | ------> | Layer Normaliza-  | ------> | Output Tensor |
| (batch_size,  |         | tion              |         | (batch_size,  |
| seq_length,   |         | (F.layer_norm)    |         | seq_length,   |
| n_embd)       |         | with weight &     |         | n_embd)       |
|               |         | bias parameters   |         |               |
+---------------+         +-------------------+         +---------------+
"""
class Normalisation(AbstractGPTModule):
    def __init__(self, n_embd, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)