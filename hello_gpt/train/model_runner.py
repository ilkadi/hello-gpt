import torch
import tiktoken
import argparse

from halo import Halo
from hello_gpt.model.model_initializer import ModelInitializer
from hello_gpt.train.config_helper import ConfigHelper
from hello_gpt.train.hardware_controller import HardwareController

class ModelRunner:
    encoding = tiktoken.get_encoding("gpt2")
    def encode(self, s): 
        return self.encoding.encode(s, allowed_special={"<|endoftext|>"})
    def decode(self, l): 
        return self.encoding.decode(l)
    
    def __init__(self, config_ext, checkpoint, gpt2base):
        config_helper = ConfigHelper()
        config = config_helper.update_with_config({}, 'hello_gpt/train/train.yaml')
        config = config_helper.update_with_config(config, 'hello_gpt/train/model_run.yaml')
        self.config = config_helper.update_with_config(config, config_ext)
        config_helper.print_config(self.config)
        
        self.device_context, self.scaler = HardwareController(self.config["data_type"], self.config["device_type"], self.config["device"]).setup_hardware()
        self.model, _ , _ = ModelInitializer(config=self.config, checkpoint=checkpoint, run_mode=True, gpt2base=gpt2base).init_model()
    
    def run_model(self):
        print("Starting the interactive console..")
        spinner = Halo(text='begging the oracle', spinner='dots')
        while True:
            user_input = input(">:")
            start_ids = self.encode(user_input)
            x = (torch.tensor(start_ids, dtype=torch.long, device=self.config["device"])[None, ...])
            spinner.start()
            with torch.no_grad():
                with self.device_context:
                    y = self.model.run_model(x, self.config['max_new_tokens'], temperature=self.config['temperature'], top_k=self.config['top_k'])
            spinner.stop()
            print(self.decode(y[0].tolist()))
            print('---------------')
            cont = input("Do you want to continue? (yes/no): ")
            if cont.lower() != 'yes':
                break

def main(config, checkpoint, gpt2base):
    runner = ModelRunner(config, checkpoint, gpt2base)
    runner.run_model()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the model run.')
    parser.add_argument('--config', type=str, required=False, help='Path to the configuration file extension (optional override).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Name of the checkpoint stored in the <out_dir>.')
    parser.add_argument('--gpt2base', type=str, required=False, help='Hugging face GPT2 base model.', default=None)
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.gpt2base)