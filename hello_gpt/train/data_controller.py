import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

"""
DataController and ModelDataset - Purpose-Oriented Flow
-------------------------------------------------------

+----------------+
| ModelDataset   |
+----------------+
| - __init__(data, block_size)                                   |
|   'Purpose: Initialize dataset with given data and block size' |
| - __len__()                                                    |
|   'Purpose: Return the length of the dataset'                  |
| - __getitem__(idx)                                             |
|   'Purpose: Retrieve a specific item from the dataset'         |
+----------------------------------------------------------------+

+----------------+
| DataController |
+----------------+
| - __init__(config, trainset, validset)                                                |
|   'Purpose: Initializes data controller with configuration and datasets'              |
|       |
|       |-- setup_data()                                                                |
|       |   'Purpose: Prepares training and validation data loaders'                    |
|       |       |
|       |       |-- Creates data directories and loads datasets                         |
|       |       |-- Constructs ModelDataset instances for training and validation       |
|       |       |-- Returns DataLoader instances for each dataset                       |
|       |
|       |-- get_batch(loader_iter)                                                      |
|           'Purpose: Retrieves a batch of data from the specified DataLoader iterator' |
|           |                                                                           |
|           |-- Uses data_to_device to move data to specified device                    |
+---------------------------------------------------------------------------------------+

* Notes:
  - ModelDataset is responsible for managing the dataset and providing data items.
  - DataController initializes the data loaders and handles batch retrieval.
  - The flow indicates the logical sequence of operations in handling and preparing data for training and validation.
"""
class ModelDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+1+self.block_size]
        return x, y

class DataController:
    data_to_device = {
        'cuda': lambda device, x, y: (x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)),
        'cpu': lambda device, x, y: (x.to(device), y.to(device))
    }
    
    def __init__(self, config, trainset, validset):
        print("Initialising the data controller..")
        self.config = config
        self.trainset = trainset
        self.validset = validset
        
        train_loader, val_loader = self.setup_data()
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        self.batch_by_dataset = {
            'train': lambda: self.get_batch(train_iter),
            'validate': lambda: self.get_batch(val_iter),
        }
    
    def setup_data(self):
        os.makedirs(self.config['out_dir'], exist_ok=True)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, 'datasets', self.config['dataset'])
        
        train_data = np.memmap(os.path.join(data_dir, self.trainset), dtype=np.uint16, mode='r').astype(np.int64)
        val_data = np.memmap(os.path.join(data_dir, self.validset), dtype=np.uint16, mode='r').astype(np.int64)
        
        train_dataset = ModelDataset(train_data, self.config['block_size'])
        val_dataset = ModelDataset(val_data, self.config['block_size'])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=True)
        return train_loader, val_loader

    def get_batch(self, loader_iter):
        x, y = next(loader_iter)
        x, y =  self.data_to_device[self.config['device_type']](self.config['device'], x, y)
        return x, y