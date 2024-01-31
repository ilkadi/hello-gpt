import torch
import unittest
from unittest import mock
import inspect
from hello_gpt.model.model import HelloGPT
from hello_gpt.model.normalisation import Normalisation

class TestHelloGPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model_config = {
            'n_layer': 2,
            'n_head': 2,
            'n_embd': 4,
            'block_size': 4,
            'vocab_size': 10,
            'dropout': 0.1,
            'bias': False,
            'dmax_flops': 1.0e12
        }
        self.model = HelloGPT(self.model_config)

    def test_initialization(self):
        # Check that the model configuration is correctly updated
        self.assertEqual(self.model.config, self.model.default_config)
        self.assertEqual(self.model.config['n_layer'], self.model_config['n_layer'])
        self.assertEqual(self.model.config['n_head'], self.model_config['n_head'])
        self.assertEqual(self.model.config['n_embd'], self.model_config['n_embd'])
        self.assertEqual(self.model.config['block_size'], self.model_config['block_size'])
        self.assertEqual(self.model.config['vocab_size'], self.model_config['vocab_size'])
        self.assertEqual(self.model.config['dropout'], self.model_config['dropout'])
        self.assertEqual(self.model.config['bias'], self.model_config['bias'])
        self.assertEqual(self.model.config['dmax_flops'], self.model_config['dmax_flops'])

        # Check that the model components are correctly initialized
        self.assertIsInstance(self.model.transformer, torch.nn.ModuleDict)
        self.assertIsInstance(self.model.transformer.wte, torch.nn.Embedding)
        self.assertIsInstance(self.model.transformer.wpe, torch.nn.Embedding)
        self.assertIsInstance(self.model.transformer.drop, torch.nn.Dropout)
        self.assertIsInstance(self.model.transformer.h, torch.nn.ModuleList)
        self.assertIsInstance(self.model.transformer.ln_f, Normalisation)
        self.assertIsInstance(self.model.lm_head, torch.nn.Linear)

    def test_forward(self):
        # Create a random tensor of shape (batch_size, sequence_length)
        idx = torch.tensor([[1, 2, 3, 4]])

        # Forward pass through the model
        logits, loss = self.model.forward(idx)

        # Check the shape of the output logits
        self.assertEqual(logits.shape, torch.Size([1, 1, self.model.config['vocab_size']]))

        # Check that the loss is None when targets are not provided
        self.assertIsNone(loss)

        # Now let's test with targets
        targets = torch.tensor([[2, 3, 4, 5]])
        logits, loss = self.model.forward(idx, targets)

        # Check the shape of the output logits
        self.assertEqual(logits.shape, torch.Size([1, 4, self.model.config['vocab_size']]))

        # Check that the loss is a scalar and its value is greater than 0
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_configure_optimizers_no_fused(self):
        weight_decay = 0.01
        learning_rate = 0.001
        betas = (0.9, 0.999)
        device_type = 'cuda'
        
        # Mock the torch.optim.AdamW function
        with mock.patch('torch.optim.AdamW', return_value='mock_optimizer') as mock_adamw:
            optimizer = self.model.configure_optimizers(weight_decay, learning_rate, betas, device_type)

        # Get the actual call arguments
        args, kwargs = mock_adamw.call_args
        self.assertIsInstance(args[0], list)
        for item in args[0]:
            self.assertIsInstance(item, dict)
            self.assertIn('params', item)
            self.assertIn('weight_decay', item)
            self.assertIsInstance(item['params'], list)
            self.assertNotEqual(len(item['params']), 0)

        # Check the other arguments
        self.assertEqual(kwargs['lr'], learning_rate)
        self.assertEqual(kwargs['betas'], betas)
        self.assertEqual(optimizer, 'mock_optimizer')
        
    def test_configure_optimizers_fused(self):
        weight_decay = 0.01
        learning_rate = 0.001
        betas = (0.9, 0.999)
        device_type = 'cuda'
        
        def custom_function(fused=True):
            return True
        mock_signature = inspect.signature(custom_function)
        
        with mock.patch('inspect.signature', return_value=mock_signature):
            with mock.patch('torch.optim.AdamW', return_value='mock_optimizer') as mock_adamw:
                optimizer = self.model.configure_optimizers(weight_decay, learning_rate, betas, device_type)

        # Get the actual call arguments
        args, kwargs = mock_adamw.call_args
        self.assertIsInstance(args[0], list)
        for item in args[0]:
            self.assertIsInstance(item, dict)
            self.assertIn('params', item)
            self.assertIn('weight_decay', item)
            self.assertIsInstance(item['params'], list)
            self.assertNotEqual(len(item['params']), 0)

        # Check the other arguments
        self.assertEqual(kwargs['lr'], learning_rate)
        self.assertEqual(kwargs['betas'], betas)
        self.assertEqual(kwargs['fused'], True)
        self.assertEqual(optimizer, 'mock_optimizer')
        
    def test_estimate_mfu(self):
        # Define the parameters for the test
        fwdbwd_per_iter = 1000
        dt_ms = 0.01

        # Call the method under test
        mfu = self.model.estimate_mfu(fwdbwd_per_iter, dt_ms)

        # Calculate the expected model flops utilisation manually
        L = self.model.config['n_layer']
        H = self.model.config['n_head']
        Q = self.model.config['n_embd'] // self.model.config['n_head']
        T = self.model.config['block_size']
        
        # 6 * self.model.n_params: Each parameter is involved in 
        # two multiplications and one addition during forward propagation, 
        # and two multiplications and one addition during backpropagation. Hence, 6 * self.model.n_params.
        # 12 * L * H * Q * T: This part of the formula accounts for the self-attention mechanism in the transformer model. 
        # For each layer (L), each attention head (H), and each token in the sequence (T), the model performs a dot product 
        # between the query vector and the key vector, both of which have a dimension of Q. 
        # This dot product involves Q multiplications and Q-1 additions. 
        # Since this is done for both forward propagation and backpropagation, we get 2 * 2 * (Q + Q - 1) = 4 * 2Q = 8Q. 
        # The remaining 4Q likely for the scaling of the dot product by 1/sqrt(Q), the addition of the value vector, and the softmax operation.
        # Note: it is an approximation assuming that the model performs the same number of operations for each token in the sequence.
        flops_per_token = 6 * self.model.n_params + 12 * L * H * Q * T
        flops_achieved = flops_per_token * T * fwdbwd_per_iter * 1000 / dt_ms
        flops_promised = float(self.model.config['dmax_flops'])
        expected_mfu = flops_achieved / flops_promised

        # Check that the calculated MFU is correct
        self.assertAlmostEqual(mfu, expected_mfu, places=5)

    def test_run_model(self):
        # Create a random tensor of shape (batch_size, sequence_length)
        idx = torch.tensor([[1, 2, 3, 4]])
        expected_output = torch.tensor([[1, 2, 3, 4, 5, 1, 1, 1, 3]])

        # Run the model
        max_new_tokens = 5
        temperature = 1.0
        top_k = 3
        output = self.model.run_model(idx, max_new_tokens, temperature, top_k)

        # print(f"output: {output} vs expected_output: {expected_output}")
        # Check the shape of the output
        self.assertEqual(output.shape, torch.Size([1, 9]))
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_run_model_different_max_new_tokens(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        max_new_tokens = 3
        temperature = 1.0
        top_k = 3
        output = self.model.run_model(idx, max_new_tokens, temperature, top_k)
        self.assertEqual(output.shape, torch.Size([1, 7]))

    def test_run_model_different_temperature(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        max_new_tokens = 5
        temperature = 0.5
        top_k = 3
        output = self.model.run_model(idx, max_new_tokens, temperature, top_k)
        self.assertEqual(output.shape, torch.Size([1, 9]))

    def test_run_model_different_top_k(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        max_new_tokens = 5
        temperature = 1.0
        top_k = 5
        output = self.model.run_model(idx, max_new_tokens, temperature, top_k)
        self.assertEqual(output.shape, torch.Size([1, 9]))
        
    def test_run_model_larger_input(self):
        idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        max_new_tokens = 5
        temperature = 1.0
        top_k = 3
        with self.assertRaises(ValueError):
            self.model.run_model(idx, max_new_tokens, temperature, top_k)
    
if __name__ == '__main__':
    unittest.main()