import unittest
from unittest.mock import patch, MagicMock
from hello_gpt.train.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.config_ext = "tests/train/resources/test_config.yaml"
        self.trainset = "train.bin"
        self.validset = "validate.bin"
        self.checkpoint = "checkpoint.pt"
        self.gpt2base = "gpt2"

        self.ch_patch = patch('hello_gpt.train.model_trainer.ConfigHelper', autospec=True)
        self.hc_patch = patch('hello_gpt.train.model_trainer.HardwareController', autospec=True)
        self.dc_patch = patch('hello_gpt.train.model_trainer.DataController', autospec=True)
        self.mc_patch = patch('hello_gpt.train.model_trainer.MonitoringController', autospec=True)
        self.mi_patch = patch('hello_gpt.train.model_trainer.ModelInitializer', autospec=True)

        self.mock_config_helper = self.ch_patch.start()
        self.mock_hardware_controller = self.hc_patch.start()
        self.mock_data_controller = self.dc_patch.start()
        self.mock_monitoring_controller = self.mc_patch.start()
        self.mock_model_initializer = self.mi_patch.start()

        self.mock_config_helper.return_value = MagicMock()
        self.mock_data_controller.return_value = MagicMock()
        self.mock_monitoring_controller.return_value = MagicMock()
        self.mock_model = MagicMock()
        self.mock_X = MagicMock()
        self.mock_Y = MagicMock()
        self.scaler = MagicMock()
        self.optimizer = MagicMock()
        mock_train_function = MagicMock(return_value=(self.mock_X, self.mock_Y))
            
        self.mock_model.return_value = ([], 2.8)
        self.mock_config_helper.return_value.update_with_config.return_value = {"data_type": "float16", 
                                                                                "device_type": "cuda", 
                                                                                "device": "cuda:0", 
                                                                                "max_iters": 100, 
                                                                                "warmup_iters": 5, 
                                                                                "lr_decay_iters": 10, 
                                                                                "min_lr": 0.01, 
                                                                                "learning_rate": 0.01, 
                                                                                "decay_lr": True, 
                                                                                "gradient_accumulation_steps": 2, 
                                                                                "grad_clip": 1.0}
        self.mock_hardware_controller.return_value.setup_hardware.return_value = (MagicMock(), self.scaler)
        self.mock_model_initializer.return_value.init_model.return_value = (self.mock_model, "model_args", self.optimizer)
        self.mock_data_controller.batch_by_dataset = {'train': mock_train_function}

    def tearDown(self):
        self.ch_patch.stop()
        self.hc_patch.stop()
        self.dc_patch.stop()
        self.mc_patch.stop()
        self.mi_patch.stop()
        
    @patch.object(ModelTrainer, 'micro_train', return_value=0.5)
    @patch.object(ModelTrainer, 'set_optimizer_lr')
    def test_train(self, mock_set_optimizer_lr, mock_micro_train):
        # Arrange
        trainer = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        trainer.monitoring_controller = self.mock_monitoring_controller
        trainer.iter_num = 0
        trainer.config = {"max_iters": 2}
        trainer.best_val_loss = 1.0

        # Act
        trainer.train()

        # Assert
        self.mock_monitoring_controller.reset_time.assert_called_once()
        self.mock_monitoring_controller.optionally_store_checkpoint.assert_called()
        self.mock_monitoring_controller.calc_model_flops_utilisation.assert_called()
        
        # Check that the methods were called the correct number of times
        self.assertEqual(mock_set_optimizer_lr.call_count, 2)
        self.assertEqual(self.mock_monitoring_controller.optionally_store_checkpoint.call_count, 2)
        self.assertEqual(mock_micro_train.call_count, 2)
        self.assertEqual(self.mock_monitoring_controller.calc_model_flops_utilisation.call_count, 2)

        # Check that the methods were called with the correct arguments
        mock_set_optimizer_lr.assert_any_call(trainer.optimizer, 0)
        mock_set_optimizer_lr.assert_any_call(trainer.optimizer, 1)
        self.mock_monitoring_controller.optionally_store_checkpoint.assert_any_call(0, 1.0)
        self.mock_monitoring_controller.optionally_store_checkpoint.assert_any_call(1, 1.0)
        self.mock_monitoring_controller.calc_model_flops_utilisation.assert_any_call(0, 0.5)
        self.mock_monitoring_controller.calc_model_flops_utilisation.assert_any_call(1, 0.5)

    def test_set_optimizer_lr(self):
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        optimizer = MagicMock()
        optimizer.param_groups = [{}]
        epoch = 50
        
        instance.set_optimizer_lr(optimizer, epoch)

        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_set_optimizer_lr_no_decay(self):
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        optimizer = MagicMock()
        optimizer.param_groups = [{}]
        
        instance.config["decay_lr"] = False
        instance.set_optimizer_lr(optimizer, 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_set_optimizer_lr_warmup(self):
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        optimizer = MagicMock()
        optimizer.param_groups = [{}]
        
        instance.set_optimizer_lr(optimizer, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.006)

    def test_set_optimizer_lr_decay(self):
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        optimizer = MagicMock()
        optimizer.param_groups = [{}]
        
        instance.set_optimizer_lr(optimizer, 7)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_set_optimizer_lr_min_lr(self):
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        optimizer = MagicMock()
        optimizer.param_groups = [{}]
        
        instance.set_optimizer_lr(optimizer, 11)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        
    @patch('torch.nn.utils.clip_grad_norm_')
    def test_micro_train_grad_clip_non_zero(self, mock_clip_grad_norm):
        # Arrange
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        instance.data_controller = self.mock_data_controller
        instance.hardware_controller = self.mock_hardware_controller
        instance.config["grad_clip"] = 0.0 

        # Act
        loss = instance.micro_train()

        # Assert
        self.mock_data_controller.batch_by_dataset['train'].assert_called()
        self.mock_model.assert_called()
        self.mock_model.assert_called_with(self.mock_X, self.mock_Y)
        self.scaler.scale.assert_called_with(loss)
        self.scaler.step.assert_called_with(self.optimizer)
        self.scaler.update.assert_called()
        self.optimizer.zero_grad.assert_called_with(set_to_none=True)
        self.assertEqual(loss, 1.4)
        mock_clip_grad_norm.assert_not_called()
        
    @patch('torch.nn.utils.clip_grad_norm_')
    def test_micro_train_grad_clip_non_zero(self, mock_clip_grad_norm):
        # Arrange
        instance = ModelTrainer(self.config_ext, self.trainset, self.validset, self.checkpoint, self.gpt2base)
        instance.data_controller = self.mock_data_controller
        instance.hardware_controller = self.mock_hardware_controller
        instance.config["grad_clip"] = 0.1 

        # Act
        loss = instance.micro_train()

        # Assert
        self.mock_data_controller.batch_by_dataset['train'].assert_called()
        self.mock_model.assert_called()
        self.mock_model.assert_called_with(self.mock_X, self.mock_Y)
        self.scaler.scale.assert_called_with(loss)
        self.scaler.step.assert_called_with(self.optimizer)
        self.scaler.update.assert_called()
        self.optimizer.zero_grad.assert_called_with(set_to_none=True)
        self.assertEqual(loss, 1.4)
        mock_clip_grad_norm.assert_called()

if __name__ == '__main__':
    unittest.main()