import torch
import os
import time

"""
MonitoringController - Purpose-Oriented Flow
---------------------------------------------

+----------------------+
| MonitoringController |
+----------------------+
| - __init__(config, data_controller, model, model_args, optimizer, device_context)   |
|   'Purpose: Initializes the monitoring controller with configurations and context'  |
|                                                                                     |
| - reset_time()                                                                      |
|   'Purpose: Resets the internal timer for performance tracking'                     |
|                                                                                     |
| - save_checkpoint(iter_num, current_loss, best_val_loss)                            |
|   'Purpose: Saves model state as a checkpoint if conditions are met'                |
|                                                                                     |
| - estimate_loss()                                                                   |
|   'Purpose: Calculates and returns the average loss over evaluation iterations'     |
|       |                                                                             |
|       |-- Utilizes data_controller for batch retrieval                              |
|       |-- Evaluates loss for each batch in 'train' and 'validate' datasets          |
|                                                                                     |
| - optionally_store_checkpoint(iter_num, best_val_loss)                              |
|   'Purpose: Evaluates and saves a checkpoint at specified intervals'                |
|       |                                                                             |
|       |-- Calls estimate_loss() to get current losses                               |
|       |-- Saves a checkpoint if criteria are met                                    |
|                                                                                     |
| - calc_model_flops_utilisation(iter_num, loss)                                      |
|   'Purpose: Calculates and logs model's floating point operations utilization'      |
|       |                                                                             |
|       |-- Computes time taken and loss for a given interval                         |
|       |-- Adjusts running mfu (model flops utilization) based on current value      |
+-------------------------------------------------------------------------------------+

* Notes:
  - The MonitoringController is responsible for performance tracking and checkpointing.
  - It interacts with the data_controller for batch processing and loss estimation.
  - Key functionalities include loss estimation, checkpoint saving, and utilization tracking.

"""
class MonitoringController:
    def __init__(self, config, data_controller, model, model_args, optimizer, device_context):
        print("Initialising the monitoring controller..")
        self.config = config
        self.data_controller = data_controller
        self.model = model
        self.model_args = model_args
        self.optimizer = optimizer
        self.device_context = device_context
        self.running_mfu = -1.0
        self.t0 = time.time()

    def reset_time(self):
        self.t0 = time.time()

    def save_checkpoint(self, iter_num, current_loss, best_val_loss):
        if current_loss < best_val_loss or self.config.always_save_checkpoint:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'model_args': self.model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': self.config,
            }
            print(f"saving checkpoint to {self.config['out_dir']}")
            torch.save(checkpoint, os.path.join(self.config['out_dir'], f"{self.config['checkpoint_name']}.pt"))

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for dataset in set(self.data_controller.batch_by_dataset.keys()):
            losses = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                X, Y = self.data_controller.batch_by_dataset[dataset]()
                with self.device_context:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[dataset] = losses.mean()
        self.model.train()
        return out
    
    def optionally_store_checkpoint(self, iter_num, best_val_loss):
        if iter_num % self.config['eval_interval'] == 0:
            losses = self.estimate_loss()
            print(f"Epoch {iter_num}: train loss {losses['train']:.4f}, validation loss {losses['validate']:.4f}")
            self.save_checkpoint(iter_num, losses['validate'], best_val_loss)

    def calc_model_flops_utilisation(self, iter_num, loss):
        t1 = time.time()
        dt = t1 - self.t0
        if iter_num % self.config['log_interval'] == 0:
            lossf = loss.item() * self.config['gradient_accumulation_steps']
            mfu = self.model.estimate_mfu(self.config['batch_size'] * self.config['gradient_accumulation_steps'], dt)
            self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
            self.t0 = t1
            
            print(f"Epoch {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%")