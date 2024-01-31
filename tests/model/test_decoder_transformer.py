import torch
import unittest
from hello_gpt.model.decoder_transformer import DecoderTransformer
from hello_gpt.model.normalisation import Normalisation
from hello_gpt.model.self_attention import SelfAttention
from hello_gpt.model.mlp import MultiLayerPerceptron

class TestDecoderTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_embd = 4
        self.n_head = 2
        self.dropout = 0.0
        self.bias = False
        self.decoder_transformer = DecoderTransformer(self.n_embd, self.n_head, self.dropout, self.bias)

    def test_initialization(self):
        # Check that all submodules have been initialized correctly
        self.assertIsInstance(self.decoder_transformer.ln_1, Normalisation)
        self.assertIsInstance(self.decoder_transformer.attn, SelfAttention)
        self.assertIsInstance(self.decoder_transformer.ln_2, Normalisation)
        self.assertIsInstance(self.decoder_transformer.mlp, MultiLayerPerceptron)

    def test_forward(self):
        # Create a static tensor of shape (batch_size, sequence_length, n_embd)
        x = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]])
        expected_output = torch.tensor([[[-0.2432,  0.2297,  0.3655,  0.4694], 
                                         [0.1568,  0.6297,  0.7655,  0.8694], 
                                         [0.5568,  1.0297,  1.1655,  1.2694]]])

        # Run the forward pass
        output = self.decoder_transformer(x)
        #print(f"output: {output} vs expected_output: {expected_output}")

        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

if __name__ == '__main__':
    unittest.main()