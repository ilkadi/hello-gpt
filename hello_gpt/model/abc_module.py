from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractGPTModule(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        """
        Define the forward pass.
        """
        pass