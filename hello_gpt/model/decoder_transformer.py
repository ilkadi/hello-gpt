from hello_gpt.model.abc_module import AbstractGPTModule
from hello_gpt.model.normalisation import Normalisation
from hello_gpt.model.self_attention import SelfAttention
from hello_gpt.model.mlp import MultiLayerPerceptron

""" Meowalization diagram for the Decoder Transformer module:
+-------------+      +----------------+      +--------------+      +----------------+      +-------------+      +--------------+
| Input       | ---> | Normalisation  | ---> | Self         | ---> | Normalisation  | ---> | Multi-Layer | ---> | Output       |
| (x)         |      | (norm_1)       |      | Attention    |      | (norm_2)       |      | Perceptron  |      | (x)          |
|             |      |                |      | (attn)       |      |                |      | (mlp)       |      |              |
+-------------+      +----------------+      +--------------+      +----------------+      +-------------+      +--------------+
    |                      |                           ^                        |                ^                  ^
    |                      |                           |                        |                |                  |
    +----------------------+-------- Residual ---------+------- Connections ----+-----------------------------------+
"""
class DecoderTransformer(AbstractGPTModule):
    def __init__(self, n_embd, n_head, dropout, bias=False):
        super().__init__()
        # Names are essential for GPT2 model loading
        self.ln_1 = Normalisation(n_embd=n_embd, bias=bias)
        self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias)
        self.ln_2 = Normalisation(n_embd=n_embd, bias=bias)
        self.mlp = MultiLayerPerceptron(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(self, x):
        normx_1 = self.ln_1(x)
        x = x + self.attn(normx_1)
        normx_2 = self.ln_2(x)
        x = x + self.mlp(normx_2)
        return x