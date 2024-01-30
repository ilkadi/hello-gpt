import torch
from contextlib import nullcontext

class HardwareController:
    torch_datatype_by_string = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    context_by_device = {
        'cuda': lambda device_type, torch_datatype: torch.amp.autocast(device_type=device_type, dtype=torch_datatype),
        'cpu': lambda device_type, torch_datatype:  nullcontext()
    }
    
    def __init__(self, data_type, device_type, device):
        self.data_type = data_type
        self.device_type = device_type
        self.device = device
        
    def setup_hardware(self):
        print("Setting up hardware..")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch_dtype = self.torch_datatype_by_string[self.data_type]
        
        device_context = self.context_by_device[self.device_type](self.device, torch_dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.data_type == 'float16'))
        return device_context, scaler