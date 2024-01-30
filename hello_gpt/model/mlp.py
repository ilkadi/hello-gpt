import torch.nn as nn
from hello_gpt.model.abc_module import AbstractGPTModule

""" Meowalization diagram for the Multi-Layer Perceptron module:
+------------+      +------------+      +------------+      +------------+
| Linear     |      | Activation |      | Linear     |      | Dropout    |
| Layer      |      | Layer      |      | Layer      |      | Layer      |
|            |      |            |      |            |      |            |
| n_embd     |      | 4*n_embd   |      | 4*n_embd   |      | n_embd     |
| to         | ---> | to         | ---> | to         | ---> |            |
| 4*n_embd   |      | 4*n_embd   |      | n_embd     |      |            |
+------------+      +------------+      +------------+      +------------+
"""
class MultiLayerPerceptron(AbstractGPTModule):
    def __init__(self, n_embd, dropout, bias=False):
        super().__init__()
        # Names are essential for GPT2 model loading
        self.c_fc     = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu     = nn.GELU()
        self.c_proj   = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x