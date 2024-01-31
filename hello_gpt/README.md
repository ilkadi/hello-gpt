# Directory overview
```
hello_gpt Project Structure - Purpose-Oriented Meowalization
------------------------------------------------------------

+-------------------+
| hello_gpt         |
| (Project Root)    |
+-------------------+
| - Purpose: Main directory containing the entire project structure. 
|
|-- datasets/
|   'Purpose: Contains dataset directories and utilities for data processing'
|       |
|       |-- edgar_poe/ : Edgar Allan Poe's works dataset
|       |-- poems/     : Collection of poems dataset
|       |-- txt2token_converter.py : Converts text to tokens for model training
|
|-- model/
|   'Purpose: Houses the core model components and initialization'
|       |
|       |-- model.py               : Main model architecture of HelloGPT
|       |-- model_initializer.py   : Initializes the HelloGPT model
|       |-- abc_module.py          : Module ABC (Abstract Base Class)
|       |-- decoder_transformer.py : Transformer decoder implementation
|       |-- mlp.py                 : Multi-layer perceptron (MLP) module
|       |-- normalisation.py       : Normalization layers for the model
|       |-- self_attention.py      : Self-attention mechanism implementation
|
|-- train/
    'Purpose: Contains training scripts, configuration, and controllers'
        |
        |-- train.py                : Main training script
        |-- model_runner.py         : Script to run the model
        |-- config_helper.py        : Helper for configuration management
        |-- data_controller.py      : Manages data loading and batching
        |-- hardware_controller.py  : Handles hardware-specific settings
        |-- monitoring_controller.py: Monitors and logs training progress
        |-- train.yaml              : YAML configuration for training
        |-- model_run.yaml          : YAML configuration for model run

* Notes:
  - The 'datasets' directory is focused on data storage and preprocessing.
  - The 'model' directory contains all the components necessary for the model's architecture.
  - The 'train' directory involves training aspects, including controllers and configurations.
```
