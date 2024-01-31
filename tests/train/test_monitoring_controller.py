import unittest
import os
from unittest.mock import patch, MagicMock
from hello_gpt.train.monitoring_controller import MonitoringController

class TestMonitoringController(unittest.TestCase):
    def setUp(self):
        self.config = {
            'out_dir': 'checkpoints',
            'checkpoint_name': 'checkpoint',
            'always_save_checkpoint': False,
            'eval_iters': 5,
            'eval_interval': 10,
            'log_interval': 20,
            'batch_size': 32,
            'gradient_accumulation_steps': 1
        }
        self.data_controller = MagicMock()
        self.model = MagicMock()
        self.model_args = MagicMock()
        self.optimizer = MagicMock()
        self.device_context = MagicMock()
        self.monitoring_controller = MonitoringController(self.config, self.data_controller, self.model, self.model_args, self.optimizer, self.device_context)

    def test_reset_time(self):
        self.monitoring_controller.reset_time()
        self.assertNotEqual(0.0, self.monitoring_controller.t0)
        
    def test_save_checkpoint_positive(self):
        with patch('torch.save') as mock_save:
            self.monitoring_controller.save_checkpoint(1, 0.5, 0.6)
            self.assertTrue(self.model.state_dict.called)
            self.assertTrue(self.optimizer.state_dict.called)
            self.assertEqual(self.model_args, self.monitoring_controller.model_args)
            self.assertEqual(self.config, self.monitoring_controller.config)
            mock_save.assert_called_once_with(
            {
                'model': self.model.state_dict(),
                'model_args': self.model_args,
                'optimizer': self.optimizer.state_dict(),
                'iter_num': 1,
                'best_val_loss': 0.5,
                'config': self.config
            },
            os.path.join(self.config['out_dir'], f"{self.config['checkpoint_name']}.pt")
            )
        
    def test_save_checkpoint_always(self):
        self.config['always_save_checkpoint'] = True
        with patch('torch.save') as mock_save:
            self.monitoring_controller.save_checkpoint(1, 0.5, 0.4)
            self.assertTrue(self.model.state_dict.called)
            self.assertTrue(self.optimizer.state_dict.called)
            self.assertEqual(self.model_args, self.monitoring_controller.model_args)
            self.assertEqual(self.config, self.monitoring_controller.config)
            mock_save.assert_called_once_with(
            {
                'model': self.model.state_dict(),
                'model_args': self.model_args,
                'optimizer': self.optimizer.state_dict(),
                'iter_num': 1,
                'best_val_loss': 0.5,
                'config': self.config
            },
            os.path.join(self.config['out_dir'], f"{self.config['checkpoint_name']}.pt")
            )

    def test_save_checkpoint_negative(self):
        self.monitoring_controller.save_checkpoint(1, 0.6, 0.5)
        self.assertFalse(self.model.state_dict.called)
        self.assertFalse(self.optimizer.state_dict.called)

    def test_estimate_loss(self):
        mock_logits = MagicMock()
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5

        # Replace the __call__ method of self.model with a mock that returns mock_logits and mock_loss
        with patch.object(type(self.monitoring_controller.model), '__call__', return_value=(mock_logits, mock_loss)):
            self.data_controller.batch_by_dataset = {
                'train': MagicMock(return_value=(MagicMock(), MagicMock())),
                'validate': MagicMock(return_value=(MagicMock(), MagicMock()))
            }
            self.monitoring_controller.model.eval = MagicMock()
            self.monitoring_controller.model.train = MagicMock()
            losses = self.monitoring_controller.estimate_loss()
            self.assertEqual(2, len(losses))
            self.assertTrue('train' in losses)
            self.assertTrue('validate' in losses)
            self.assertTrue(self.monitoring_controller.model.eval.called)
            self.assertTrue(self.monitoring_controller.model.train.called)

    def test_optionally_store_checkpoint_positive(self):
        self.monitoring_controller.estimate_loss = MagicMock(return_value={'train': 0.5, 'validate': 0.4})
        self.monitoring_controller.save_checkpoint = MagicMock()
        self.monitoring_controller.optionally_store_checkpoint(10, 0.3)
        self.assertTrue(self.monitoring_controller.estimate_loss.called)
        self.assertTrue(self.monitoring_controller.save_checkpoint.called)

    def test_optionally_store_checkpoint_negative(self):
        self.monitoring_controller.estimate_loss = MagicMock(return_value={'train': 0.5, 'validate': 0.6})
        self.monitoring_controller.save_checkpoint = MagicMock()
        self.monitoring_controller.optionally_store_checkpoint(1, 0.3)
        self.assertFalse(self.monitoring_controller.estimate_loss.called)
        self.assertFalse(self.monitoring_controller.save_checkpoint.called)

    def test_calc_model_flops_utilisation_positive(self):
        self.monitoring_controller.model.estimate_mfu = MagicMock(return_value=0.8)
        self.monitoring_controller.t0 = 0.0
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        self.monitoring_controller.calc_model_flops_utilisation(20, mock_loss)
        self.assertNotEqual(0.0, self.monitoring_controller.t0)

    def test_calc_model_flops_utilisation_negative(self):
        self.monitoring_controller.model.estimate_mfu = MagicMock(return_value=0.8)
        self.monitoring_controller.t0 = 0.0
        self.monitoring_controller.calc_model_flops_utilisation(1, MagicMock(item=0.5))
        self.assertFalse(self.monitoring_controller.model.estimate_mfu.called)
        self.assertEqual(0.0, self.monitoring_controller.t0)

if __name__ == '__main__':
    unittest.main()