import os
import pprint
import torch
from hello_gpt.model.model import HelloGPT

"""
ModelInitializer Class - Purpose-Oriented Meowalization
-------------------------------------------------------

+------------------------+
| ModelInitializer Class |
+------------------------+
       |
       |-- __init__(config, checkpoint, run_mode, gpt2base)
       |    'Purpose: Initialize configuration and state parameters'
       |
       |-- init_model()
       |    'Purpose: Orchestrates model setup and optimization'
       |       |
       |       |-- setup_model(model_args)
       |       |    'Purpose: Configures model from scratch or based on a checkpoint'
       |       |       |
       |       |       |-- from_pretrained(gpt2base, model_args)
       |       |       |    'Purpose: Adapts a GPT2 model to HelloGPT format'
       |       |       |
       |       |       |-- HelloGPT(model_args)
       |       |       |    'Purpose: Creates a new custom GPT model if no pretraining'
       |       |
       |       |    'Decides mode (training/evaluation), sets device, and initializes optimizer'
       |
       |-- setup_model(model_args)
       |    'Purpose: Loads and modifies model with checkpoint data if available'
       |       |
       |       |-- Loads model checkpoint
       |       |    'Purpose: Retrieves stored model state for continuity'
       |       |
       |       |-- from_pretrained() / HelloGPT()
       |       |    'Purpose: Ensures model aligns with specified configuration'
       |
       |-- from_pretrained(gpt2base, model_args)
            'Purpose: Transforms a GPT2 model to match HelloGPT specifications'
"""
class ModelInitializer:
      
    def __init__(self, config, checkpoint,  run_mode=False, gpt2base=None):
        self.config = config
        self.checkpoint = checkpoint
        self.run_mode = run_mode
        self.gpt2base = gpt2base
    
    def init_model(self):
        print("Initialising the model..")
        model_args = dict(n_layer=self.config["n_layer"], 
                          n_head=self.config["n_head"], 
                          n_embd=self.config["n_embd"], 
                          block_size=self.config["block_size"],
                          bias=self.config["bias"], 
                          vocab_size=self.config["vocab_size"], 
                          dropout=self.config["dropout"],
                          dmax_flops=self.config["dmax_flops"]) 
        model, model_args, torch_checkpoint = self.setup_model(model_args)
        optimizer = None
        
        if self.run_mode:
            model.eval()
            model.to(self.config["device"])
        else:
            model.to(self.config["device"])
            optimizer = model.configure_optimizers(self.config["weight_decay"], 
                                   self.config["learning_rate"],
                                   (self.config["beta1"], self.config["beta2"]),
                                   self.config["device"])
            if self.checkpoint:
                optimizer.load_state_dict(torch_checkpoint['optimizer'])
        
        print("Model initialised with the following configuration:")
        pprint.pprint(model_args)   
        return model, model_args, optimizer

    def setup_model(self, model_args):
        model = model = self.from_pretrained(self.gpt2base, model_args) if self.gpt2base else HelloGPT(model_args)
        torch_checkpoint = None
        
        if self.checkpoint:
            print(f"Initialising the model from checkpoint {self.checkpoint} ..")
            ckpt_path = os.path.join(self.config['out_dir'], self.checkpoint)
            torch_checkpoint = torch.load(ckpt_path, map_location=self.config["device"])
            
            if self.run_mode:
                state_dict = torch_checkpoint['model']
                model.load_state_dict(state_dict)
            else:
                checkpoint_model_args = torch_checkpoint['model_args']    
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                    model_args[k] = checkpoint_model_args[k]
        return model, model_args, torch_checkpoint
    
    def from_pretrained(self, gpt2base, model_args):
        supported_gpts = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }
        assert gpt2base in supported_gpts.keys()
        
        from transformers import GPT2LMHeadModel
        print("GPT2 enforcing: vocab_size=50257, block_size=1024, bias=True")
        model_args.update(supported_gpts[gpt2base])
        model_args['vocab_size'] = 50257
        model_args['block_size'] = 1024
        model_args['bias'] = True
        
        model_hg = HelloGPT(model_args)
        hello_gpt_sd = model_hg.state_dict()
        hello_gpt_sd_keys = hello_gpt_sd.keys()
        hello_gpt_sd_keys = [k for k in hello_gpt_sd_keys if not k.endswith('.attn.bias')]
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(gpt2base)
        hug_gpt_sd = model_hf.state_dict()
        hug_gpt_sd_keys = hug_gpt_sd.keys()
        hug_gpt_sd_keys = [k for k in hug_gpt_sd_keys if not k.endswith('.attn.masked_bias')] 
        hug_gpt_sd_keys = [k for k in hug_gpt_sd_keys if not k.endswith('.attn.bias')] 
        
        # Conv1D to Linear
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(hug_gpt_sd_keys) == len(hello_gpt_sd_keys), f"mismatched keys: {len(hug_gpt_sd_keys)} != {len(hello_gpt_sd_keys)}"
        
        print(f"Converting from {gpt2base} into hello-GPT..")
        for k in hug_gpt_sd_keys:
            if any(k.endswith(w) for w in transposed):
                # Conv1D weights to transpose
                # print(f"Transposing {k}: hug {hug_gpt_sd[k].shape[::-1]} into hello {hello_gpt_sd[k].shape} ..")
                assert hug_gpt_sd[k].shape[::-1] == hello_gpt_sd[k].shape
                with torch.no_grad():
                    hello_gpt_sd[k].copy_(hug_gpt_sd[k].t())
            else:
                # Linear copy over the rest
                # print(f"Copying {k}: hug {hug_gpt_sd[k].shape} into hello {hello_gpt_sd[k].shape} ..")
                assert hug_gpt_sd[k].shape == hello_gpt_sd[k].shape
                with torch.no_grad():
                    hello_gpt_sd[k].copy_(hug_gpt_sd[k])

        return model_hg