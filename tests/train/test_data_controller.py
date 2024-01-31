import unittest
import os
import torch
from hello_gpt.train.data_controller import DataController

class TestDataController(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Create a dummy configuration
        config = {
            'out_dir': 'checkpoints',
            'dataset': 'unittests',
            'block_size': 10,
            'batch_size': 32,
            'device_type': 'cuda',
            'device': 'cuda:0'
        }

        # Create dummy trainset and validset filenames
        trainset = 'test_train.bin'
        validset = 'test_validate.bin'

        # Create an instance of DataController
        self.data_controller = DataController(config, trainset, validset)
        
    def test_init_controller(self):
        # Check if the data directories are created
        self.assertTrue(os.path.exists(self.data_controller.config['out_dir']))

        # Check if the train and validation datasets are loaded correctly
        self.assertIsNotNone(self.data_controller.trainset)
        self.assertIsNotNone(self.data_controller.validset)

        # Check if the train and validation loaders are created
        self.assertIsNotNone(self.data_controller.batch_by_dataset['train'])
        self.assertIsNotNone(self.data_controller.batch_by_dataset['validate'])

    def test_get_batch(self):
        # Get a batch of data
        x, y = self.data_controller.batch_by_dataset['train']()

        # Check if the data is moved to the correct device
        self.assertEqual(x.device.type, self.data_controller.config['device_type'])
        self.assertEqual(y.device.type, self.data_controller.config['device_type'])

        # Check if the batch size is correct
        self.assertEqual(x.size(0), self.data_controller.config['batch_size'])
        self.assertEqual(y.size(0), self.data_controller.config['batch_size'])

if __name__ == '__main__':
    unittest.main()