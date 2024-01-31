import torch
import unittest
from hello_gpt.model.normalisation import Normalisation

class TestNormalisation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_embd = 2
        self.norm = Normalisation(self.n_embd)

    def test_initialization(self):
        # Check that the weight and bias parameters are initialized correctly
        self.assertTrue(torch.allclose(self.norm.weight, torch.ones(self.n_embd)))
        self.assertIsNone(self.norm.bias)

    def test_forward(self):
        # Create a random tensor of shape (batch_size, sequence_length, n_embd)
        x = torch.tensor([[[0.1, 0.5]]])
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        expected_output = (x - mean) / torch.sqrt(var + 1e-5)
        
        output = self.norm(x).detach()
        #print(f"output: {output} vs expected_output: {expected_output}")

        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, x.shape)        
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

if __name__ == '__main__':
    unittest.main()