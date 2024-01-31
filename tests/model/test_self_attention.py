import torch
import unittest
import math
from torch.nn import functional as F
from hello_gpt.model.self_attention import SelfAttention

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_embd = 4
        self.n_head = 2
        self.dropout = 0 # using 0 to keep flash and manual impementations in sync
        self.self_attn = SelfAttention(self.n_embd, self.n_head, self.dropout)

    def test_initialization(self):
        # Check that the linear layers are initialized with the correct dimensions
        self.assertEqual(self.self_attn.c_attn.in_features, self.n_embd)
        self.assertEqual(self.self_attn.c_attn.out_features, 3 * self.n_embd)
        self.assertEqual(self.self_attn.c_proj.in_features, self.n_embd)
        self.assertEqual(self.self_attn.c_proj.out_features, self.n_embd)

        # Check that the dropout layers are initialized with the correct dropout rate
        self.assertEqual(self.self_attn.attn_dropout.p, self.dropout)
        self.assertEqual(self.self_attn.resid_dropout.p, self.dropout)

    def test_forward(self):
        self.self_attn.register_buffer("bias", torch.tril(torch.ones(3, 3))
                                        .view(1, 1, 3, 3))
        
        # Create a static tensor of shape (batch_size, sequence_length, n_embd)
        x = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]])

        # Manually calculate the forward pass
        batch_size, seq_length, n_embd = x.size() 
        q, k, v = self.self_attn.c_attn(x).split(n_embd, dim=2)
        k = k.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, n_embd // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.self_attn.bias[:,:,:seq_length,:seq_length] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.self_attn.attn_dropout(att)
        y_expected = att @ v
        y_expected = y_expected.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embd)
        y_expected = self.self_attn.resid_dropout(self.self_attn.c_proj(y_expected))

        # Run the forward pass
        output = self.self_attn(x)
        #print(f"output: {output} vs expected_output: {y_expected}")

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.allclose(output, y_expected, atol=1e-4))
        
if __name__ == '__main__':
    unittest.main()