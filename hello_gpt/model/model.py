import math
import pprint
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F
from hello_gpt.model.normalisation import Normalisation
from hello_gpt.model.decoder_transformer import DecoderTransformer

"""
GPT-2 Model Structure
-----------------------------
  ___________ 
 |           |       ____________________
 |   wte     |---->| Embedding Layer     |
 |___________|     | - Vocab Size        |
     |             | - Embedding Dim     |
     |             |_____________________|
     |                
     |             ____________________
     |            | Embedding Layer     |
     |----->| wpe | - Block Size        |
     |            | - Embedding Dim     |
     |            |_____________________|
     |
     |             ____________________
     |            | Dropout Layer       |
     |----->| drop| - Dropout Rate      |
     |            |_____________________|
     |
     |             ____________________       ____________________
     |            | Decoder Layer 1    |     | Decoder Layer N    |
     |            | (Variable Layers)  | ... | (N from Config)    |
     |----->|  h  | - Embedding Dim    |     | - Embedding Dim    |
     |            | - Number of Heads  |     | - Number of Heads  |
     |            | - Dropout          |     | - Dropout          |
     |            | - Bias             |     | - Bias             |
     |            |____________________|     |____________________|
     |
     |             ____________________
     |----->| ln_f| Normalisation      |
            |     | - Embedding Dim    |
            |     | - Bias             |
            |     |____________________|
            |
            |       ____________________
            |----->| Linear Layer       |
                   | - Output Vocab Size|
                   | - Embedding Dim    |
                   |____________________|

 * Note: wte.weight and lm_head.weight are shared.
"""
class HelloGPT(nn.Module):
    default_config = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 1024,
        'vocab_size': 50304,
        'dropout': 0.0,
        'bias': False,
        'dmax_flops': 1.0e12 # 1 TFLOP
    }
    
    def __init__(self, model_config):
        super().__init__()
        print("Initialising GPT model ...")
        self.default_config.update(model_config)
        self.config = self.default_config
        pprint.pprint(f"Actual model config: {self.config}")
        
        # transformer name is essential for hugging face GPT2 compatibility
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config['vocab_size'], self.config['n_embd']),
            wpe = nn.Embedding(self.config['block_size'], self.config['n_embd']),
            drop = nn.Dropout(self.config['dropout']),
            h = nn.ModuleList([DecoderTransformer(n_embd=self.config['n_embd'], n_head=self.config['n_head'], dropout=self.config['dropout'], bias=self.config['bias']) for _ in range(self.config['n_layer'])]),
            ln_f = Normalisation(self.config['n_embd'], bias=self.config['bias']),
        ))
        self.lm_head = nn.Linear(self.config['n_embd'], self.config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)           
        self.n_params = sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()
        print("total params: %.2fM" % (self.n_params/1e6,))

    # init weights per GPT-2 paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        for pn, p in self.named_parameters():
            if pn.endswith('combined_proj.weight'): # hacky init of combined projection at self-attention blocks
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config['n_layer']))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['block_size'], f"Cannot forward sequence of length {t}, block size is only {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed tensors: {len(decay_params)}, with {num_decay_params:,} params total")
        print(f"Non-decayed tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params total")
        
        # Use AdamW optimizer if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"AdamW optimizer enabled: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        L, H, Q, T = self.config['n_layer'], self.config['n_head'], self.config['n_embd']//self.config['n_head'], self.config['block_size']
        flops_per_token = 6*self.n_params + 12*L*H*Q*T
        flops_achieved = flops_per_token * T * fwdbwd_per_iter /dt 
        flops_promised = float(self.config['dmax_flops'])
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def run_model(self, idx, max_new_tokens, temperature, top_k):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx