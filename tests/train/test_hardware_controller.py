import unittest
from unittest import mock
from hello_gpt.train.hardware_controller import HardwareController

class TestHardwareController(unittest.TestCase):
    def setUp(self):
        self.data_type = 'float32'
        self.device_type = 'cuda'
        self.device = 'cuda:0'
        self.controller = HardwareController(self.data_type, self.device_type, self.device)

    def test_init_controller(self):
        with mock.patch('torch.amp.autocast'):
            self.assertIsNotNone(self.controller.data_type)
            self.assertIsNotNone(self.controller.device_type)
            self.assertIsNotNone(self.controller.device)
        
    def test_setup_hardware(self):
        with mock.patch('torch.cuda.amp.GradScaler') as mock_grad_scaler, \
             mock.patch('torch.amp.autocast') as mock_autocast:

            mock_grad_scaler.return_value = mock.Mock()
            mock_autocast.return_value = mock.Mock()

            device_context, scaler = self.controller.setup_hardware()

            self.assertEqual(mock_grad_scaler.call_count, 1)
            self.assertEqual(mock_autocast.call_count, 1)
            self.assertIsInstance(device_context, mock.Mock)
            self.assertIsInstance(scaler, mock.Mock)

if __name__ == '__main__':
    unittest.main()