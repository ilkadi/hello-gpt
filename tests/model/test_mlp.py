import torch
import unittest
from hello_gpt.model.mlp import MultiLayerPerceptron

class TestMultiLayerPerceptron(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_embd = 2
        self.dropout = 0.1
        self.mlp = MultiLayerPerceptron(self.n_embd, self.dropout)

    def test_initialization(self):
        # Check that the linear layers are initialized with the correct dimensions
        self.assertEqual(self.mlp.c_fc.in_features, self.n_embd)
        self.assertEqual(self.mlp.c_fc.out_features, 4 * self.n_embd)
        self.assertEqual(self.mlp.c_proj.in_features, 4 * self.n_embd)
        self.assertEqual(self.mlp.c_proj.out_features, self.n_embd)

        # Check that the dropout layer is initialized with the correct dropout rate
        self.assertEqual(self.mlp.dropout.p, self.dropout)

    def test_forward(self):
        # Create a random tensor of shape (batch_size, sequence_length, n_embd)
        x = torch.tensor([[[0.1, 0.5]]])
        expected_output = torch.tensor([[[0.0837, -0.1383]]])
        output = self.mlp(x).detach()  
        # print(f"output: {output} vs expected_output: {expected_output}")

        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

if __name__ == '__main__':
    unittest.main()