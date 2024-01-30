import math
import torch
import argparse

from hello_gpt.model.model_initializer import ModelInitializer
from hello_gpt.train.config_helper import ConfigHelper
from hello_gpt.train.hardware_controller import HardwareController
from hello_gpt.train.data_controller import DataController
from hello_gpt.train.monitoring_controller import MonitoringController
    
class ModelTrainer:
    
    def __init__(self, config_ext, trainset, validset, checkpoint, gpt2base):
        torch.manual_seed(89237410)
        self.iter_num = 0
        self.best_val_loss = 1e9
        
        config_helper = ConfigHelper()
        config = config_helper.update_with_config({}, 'hello_gpt/train/train.yaml')
        self.config = config_helper.update_with_config(config, config_ext)
        config_helper.print_config(self.config)
        self.data_controller = DataController(self.config, trainset, validset)
        
        self.device_context, self.scaler = HardwareController(self.config["data_type"], self.config["device_type"], self.config["device"]).setup_hardware()
        self.model, self.model_args, self.optimizer = ModelInitializer(self.config, checkpoint, gpt2base=gpt2base).init_model()
        self.monitoring_controller = MonitoringController(self.config, self.data_controller, self.model, self.model_args, self.optimizer, self.device_context)

    def train(self):
        print("Starting the training process..")
        self.monitoring_controller.reset_time()
        
        best_val_loss = self.best_val_loss
        max_iters = self.config["max_iters"] + self.iter_num
        
        for iter_num in range(self.iter_num, max_iters):
            self.set_optimizer_lr(self.optimizer, iter_num)
            self.monitoring_controller.optionally_store_checkpoint(iter_num, best_val_loss)
            loss = self.micro_train()
            self.monitoring_controller.calc_model_flops_utilisation(iter_num, loss)

    def set_optimizer_lr(self, optimizer, epoch):
        if not self.config["decay_lr"]:
            lr = self.config["learning_rate"]
        elif epoch < self.config["warmup_iters"]:
            lr = self.config["learning_rate"] * epoch / self.config["warmup_iters"]
        elif epoch > self.config["lr_decay_iters"]:
            lr = self.config["min_lr"]
        else:
            decay_ratio = (epoch - self.config["warmup_iters"]) / (self.config["lr_decay_iters"] - self.config["warmup_iters"])
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.config["min_lr"] + coeff * (self.config["learning_rate"] - self.config["min_lr"])
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def micro_train(self):        
        X, Y = self.data_controller.batch_by_dataset['train']()
        for _ in range(self.config["gradient_accumulation_steps"]):
            with self.device_context:
                logits, loss = self.model(X, Y)
                loss = loss / self.config["gradient_accumulation_steps"]
        self.scaler.scale(loss).backward()
        
        if self.config["grad_clip"] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return loss

def main(config, trainset, validset, checkpoint, gpt2base):
    trainer = ModelTrainer(config, trainset, validset, checkpoint, gpt2base)
    trainer.train()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the training process.')
    parser.add_argument('--config', type=str, required=False, help='Path to the configuration file extension (optional override).')
    parser.add_argument('--trainset', type=str, required=False, help='Path to the training dataset.', default='train.bin')
    parser.add_argument('--validset', type=str, required=False, help='Path to the validation dataset.', default='validate.bin')
    parser.add_argument('--checkpoint', type=str, required=False, help='Name of the checkpoint stored in the <out_dir>.')
    parser.add_argument('--gpt2base', type=str, required=False, help='Hugging face GPT2 base model.', default=None)
    args = parser.parse_args()

    main(args.config, args.trainset, args.validset, args.checkpoint, args.gpt2base)