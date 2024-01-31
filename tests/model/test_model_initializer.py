import unittest
import os
import pytest
from unittest.mock import patch, MagicMock
from hello_gpt.model.model_initializer import ModelInitializer
from hello_gpt.model.model import HelloGPT

class TestModelInitializer(unittest.TestCase):
    def setUp(self):
        self.config = {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
            "bias": True,
            "vocab_size": 50257,
            "dropout": 0.1,
            "dmax_flops": 1000,
            "device": "cuda",
            "out_dir": "/path/to/checkpoints",
            "weight_decay": 0.01,
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999
        }
        self.checkpoint = None
        self.run_mode = False
        self.gpt2base = None

    def test_init_model(self):
        model_initializer = ModelInitializer(self.config, None, True)
        model, model_args, optimizer = model_initializer.init_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(model_args)
        self.assertIsNone(optimizer)

    def test_init_model_no_checkpoint(self):
        model_initializer = ModelInitializer(self.config, None)
        model, model_args, optimizer = model_initializer.init_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(model_args)
        self.assertIsNotNone(optimizer) 

    def test_init_model_with_checkpoint(self):
        with patch('torch.load') as mock_load, patch.object(HelloGPT, 'configure_optimizers') as mock_optim:
            mock_load.return_value = {
                'model': ...,
                'model_args': {
                    'n_layer': 12,
                    'n_head': 12,
                    'n_embd': 768,
                    'block_size': 1024,
                    'bias': True,
                    'vocab_size': 50257
                },
                'optimizer': {}
            }
            
            mock_optimizer = MagicMock()
            mock_optimizer.load_state_dict = MagicMock()
            mock_optim.return_value = mock_optimizer

            model_initializer = ModelInitializer(self.config, 'checkpoint.pt')
            model, model_args, optimizer = model_initializer.init_model()

            self.assertIsNotNone(model)
            self.assertIsNotNone(model_args)
            self.assertIsNotNone(optimizer)  # With a checkpoint, an optimizer should be loaded

            mock_load.assert_called_once_with(os.path.join(self.config['out_dir'], 'checkpoint.pt'), map_location=self.config["device"])
            mock_optim.assert_called_once_with(self.config["weight_decay"], 
                                            self.config["learning_rate"],
                                            (self.config["beta1"], self.config["beta2"]),
                                            self.config["device"])
            mock_load.assert_called_once_with(os.path.join(self.config['out_dir'], 'checkpoint.pt'), map_location=self.config["device"])
            mock_optimizer.load_state_dict.assert_called_once_with(mock_load.return_value['optimizer'])
            
    def test_setup_with_checkpoint_model_run_mode(self):
        with patch('torch.load') as mock_torch_load, patch.object(HelloGPT, 'load_state_dict'):
            mock_torch_load.return_value = {
                'model': {"mock": "model"},
                'model_args': {},
                'optimizer': {}
            }
            
            model_initializer = ModelInitializer(self.config, 'checkpoint.pt', True)
            model_args = {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
                "bias": True,
                "vocab_size": 50257
            }
            model, model_args, torch_checkpoint = model_initializer.setup_model(model_args)
            self.assertIsNotNone(model)
            self.assertIsNotNone(model_args)
            self.assertIsNotNone(torch_checkpoint)
            self.assertEqual(torch_checkpoint, mock_torch_load.return_value)
            model.load_state_dict.assert_called_once_with(torch_checkpoint['model'])
        
    def test_setup_model_with_checkpoint_train_mode(self):
        with patch('torch.load') as mock_load, patch.object(HelloGPT, 'configure_optimizers', return_value='optimizer') as mock_optim:
            mock_load.return_value = {
                'model': {},
                'model_args': {
                    'n_layer': 12,
                    'n_head': 12,
                    'n_embd': 768,
                    'block_size': 1024,
                    'bias': True,
                    'vocab_size': 50257
                },
                'optimizer': {}
            }
            model_initializer = ModelInitializer(self.config, 'checkpoint.pt', False)
            model_args = {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
                "bias": True,
                "vocab_size": 50257
            }
            model, model_args, torch_checkpoint = model_initializer.setup_model(model_args)
            self.assertIsNotNone(model)
            self.assertIsNotNone(model_args)
            self.assertIsNotNone(torch_checkpoint)  # In train mode, torch_checkpoint should not be None
        
    def test_setup_model_without_checkpoint(self):
        model_initializer = ModelInitializer(self.config, None)
        model_args = {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
            "bias": True,
            "vocab_size": 50257
        }
        model, model_args, torch_checkpoint = model_initializer.setup_model(model_args)

        self.assertIsNotNone(model)
        self.assertIsNotNone(model_args)
        self.assertIsNone(torch_checkpoint)

    def test_setup_model_from_pretrained(self):
        with patch.object(ModelInitializer, 'from_pretrained') as mock_pretrain:
            model_initializer = ModelInitializer(self.config, None, False, 'gpt2')
            model_args = {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
                "bias": True,
                "vocab_size": 50257
            }
            
            mock_model = MagicMock()
            mock_pretrain.return_value = mock_model
            
            model_hg, _, _ = model_initializer.setup_model(model_args)

            self.assertEqual(model_hg, mock_model)
            mock_pretrain.assert_called_once_with('gpt2', model_args)

    def test_from_pretrained_unrecognized_gpt2base(self):
        model_initializer = ModelInitializer(self.config, None, True)
        with pytest.raises(ValueError):
            model_initializer.from_pretrained('unrecognized', {})

    def test_from_pretrained_loads_model_integration(self):
        model_initializer = ModelInitializer(self.config, None, True)
        model_args = {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
            "bias": True,
            "vocab_size": 50257
        }
        model = model_initializer.from_pretrained('gpt2', model_args)
        assert model is not None

if __name__ == '__main__':
    unittest.main()