import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
from contextlib import nullcontext
from hello_gpt.train.model_runner import ModelRunner
from hello_gpt.model.model_initializer import ModelInitializer
from hello_gpt.train.config_helper import ConfigHelper
from hello_gpt.train.hardware_controller import HardwareController

class TestModelRunner(unittest.TestCase):
    def setUp(self):
        self.config_ext = "tests/train/resources/test_config.yaml"
        self.checkpoint = "checkpoint.pt"
        self.gpt2base = "gpt2"

        self.ch_patch = patch('hello_gpt.train.model_runner.ConfigHelper', autospec=True)
        self.hc_patch = patch('hello_gpt.train.model_runner.HardwareController', autospec=True)
        self.mi_patch = patch('hello_gpt.train.model_runner.ModelInitializer', autospec=True)

        self.mock_config_helper = self.ch_patch.start()
        self.mock_hardware_controller = self.hc_patch.start()
        self.mock_model_initializer = self.mi_patch.start()

        self.mock_config_helper.return_value = MagicMock()
        self.mock_hardware_controller.return_value = MagicMock()
        self.mock_model_initializer.return_value = MagicMock()
        self.mock_model = MagicMock()
            
        self.mock_config_helper.return_value.update_with_config.return_value = {"data_type": "float16", 
                                                                                "device_type": "cuda", 
                                                                                "device": "cuda:0", 
                                                                                "max_new_tokens": 1, 
                                                                                "temperature": 1.0, 
                                                                                "top_k": 1, 
                                                                                "out_dir": "checkpoints"}
        self.mock_hardware_controller.return_value.setup_hardware.return_value = (MagicMock(spec=nullcontext), "scaler")
        self.mock_model_initializer.return_value.init_model.return_value = (self.mock_model, "model_args", "optimizer")

    def tearDown(self):
        self.ch_patch.stop()
        self.hc_patch.stop()
        self.mi_patch.stop()

    def test_init_all_parameters_provided(self):
        # Act
        instance = ModelRunner(self.config_ext, self.checkpoint, self.gpt2base)

        # Assert
        self.mock_config_helper.assert_called_once()
        self.mock_hardware_controller.assert_called_once_with(instance.config["data_type"], instance.config["device_type"], instance.config["device"])
        self.mock_model_initializer.assert_called_once_with(config=instance.config, checkpoint=self.checkpoint, run_mode=True, gpt2base=self.gpt2base)

    def test_init_hugging_parameters_provided(self):
        # Act
        instance = ModelRunner(None, None, self.gpt2base)

        # Assert
        self.mock_config_helper.assert_called_once()
        self.mock_hardware_controller.assert_called_once_with(instance.config["data_type"], instance.config["device_type"], instance.config["device"])
        self.mock_model_initializer.assert_called_once_with(config=instance.config, checkpoint=None, run_mode=True, gpt2base=self.gpt2base)
        
    def test_init_hugging_parameters_provided(self):
        # Act
        instance = ModelRunner(self.config_ext, self.checkpoint, None)

        # Assert
        self.mock_config_helper.assert_called_once()
        self.mock_hardware_controller.assert_called_once_with(instance.config["data_type"], instance.config["device_type"], instance.config["device"])
        self.mock_model_initializer.assert_called_once_with(config=instance.config, checkpoint=self.checkpoint, run_mode=True, gpt2base=None)

    def test_init_no_model_specified(self):
        # Act and Assert
        with self.assertRaises(ValueError):
            ModelRunner(self.config_ext)
            
    def test_init_no_config_for_custom_model(self):
        # Act and Assert
        with self.assertRaises(ValueError):
            ModelRunner(None, self.checkpoint, None)
            
    def test_encode(self):
        runner = ModelRunner(self.config_ext, self.checkpoint, self.gpt2base)
        encoded = runner.encode("Hello, world!")
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(token, int) for token in encoded))
        self.assertEqual(encoded, [15496, 11, 995, 0])

    def test_decode(self):
        runner = ModelRunner(self.config_ext, self.checkpoint, self.gpt2base)
        decoded = runner.decode([15496, 11, 995, 0])
        self.assertIsInstance(decoded, str)
        self.assertEqual(decoded, "Hello, world!")

    @patch('builtins.input', side_effect=["input1", "yes", "input2", "no"])
    @patch('sys.stdout', new_callable=StringIO)
    def test_run_model(self, mock_stdout, mock_input):
        runner = ModelRunner(self.config_ext, self.checkpoint, self.gpt2base)
        runner.run_model()
        output = mock_stdout.getvalue()
        self.assertIn("Starting the interactive console..", output)
        # cannot assert inputs
        self.mock_model.run_model.assert_called()
        self.assertEqual(self.mock_model.run_model.call_count, 2)

if __name__ == '__main__':
    unittest.main()