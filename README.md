# Hello-GPT

Hello-GPT is a project that implements a simple version of the GPT model. 
It's designed to experiment with language models in a hands-on way.

## Acknowledgements

This project wouldn't be possible without the kind heroes sharing their wisdom:
* [LLM-course](https://github.com/mlabonne/llm-course?tab=readme-ov-file) by Maxime Labonne was a structured starting point
* [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrey Karpathy 
* [3BlueBrown](https://www.youtube.com/@3blue1brown) is the go-to place to recall the Math and feel it alive
* [Project Gutenberg](https://www.gutenberg.org/) for providing free access to books
* [Meowlizer](https://chat.openai.com/g/g-WlIrNB3KH-meowalizer) a great assistant to generate ASCII diagrams used throughout the project

## Project Structure

```
Simplified Project Structure Meowalization
------------------------------------------
ModelRunner --> ModelInitializer --> HelloGPT
                              |
                              |-> ConfigHelper
                              |-> HardwareController

ModelTrainer --> ModelInitializer --> HelloGPT
                              |
                              |-> ConfigHelper
                              |-> HardwareController
                              |-> DataController --> ModelDataset
                              |
                              |-> MonitoringController

* Notes:
  - Both "ModelRunner" and "ModelTrainer" depend on "ModelInitializer", "ConfigHelper", and "HardwareController".
  - "ModelInitializer" leads to "HelloGPT", representing a direct dependency.
  - "ModelTrainer" has additional dependencies on "DataController" (which in turn depends on "ModelDataset") and "MonitoringController".
  - This layout focuses on the class-level relationships, making it clearer how each component interacts within the project.
```

The project is organized into several directories:

- `checkpoints/`: The target directory for pre-trained model weights.
- `hello_gpt/`: Contains the main source code for the project.
  - `datasets/`: Contains the datasets used for training and testing the model.
  - `model/`: Contains the implementation of the GPT model and its components.
  - `train/`: Contains scripts and configuration files for training the model.
- `requirements.txt`: Lists the Python dependencies required by the project.
- `setup.py`: Script for packaging the project.

## Installation

To use 'cuda' device, follow [NVIDIA Docs](https://developer.nvidia.com/cuda-downloads).
And install a correct version of [PyTorch](https://pytorch.org/).

To install the project's dependencies, run the following command:
```
pip install -r requirements.txt
```
Note that there are further optimisations available for newer GPUs but not implemented in this code.

## How to Run
The project offers `train`, `tune` and `run` modes.
For all of those modes a configuration needs to be specified. 

### Run mode
From the root directory run the following command:
```
python -m hello_gpt.train.model_runner --checkpoint="poems_hugs.pt" --gpt2base="gpt2"
```
Here we specify the checkpoint name (in `checkpoints` directory) and we unfortunately need to add a custom flag to notify the code that it is to follow a special model initialization process based on the hugging face implementation of GPT2.

Under the hood, the `model_runner` script would load `train/train.yaml` config, update it with `model_run.yaml` contents and (optionaly) update with any other config supplied via `--config` flag. Please note that checkpoints are not included as those are big files (500MB+).

Running the above command would execute the smallest GPT2 model tuned on the poems dataset (`poems_hugs.pt` checkpoint):
```
Starting the interactive console..
>:On crimson dawn I left the town
On crimson dawn I left the town and went on to the house of the young woman in the snow. When she had gone she gave me my knapsack and said, "This is my house." I sat in the door and made a prayer. She went away and said to me, "You are a stranger to me." I replied, "Yes, my Lord." "Oh! my Lord! have you not heard of an ancient house?"
---------------
Do you want to continue? (yes/no): yes
>:Tender petals of jasmine
Tender petals of jasmine balsamic

The herb has been modified to have a reddish colour, making it paler.

The water-resistant material has been used to protect the herb against the cold and damp weather.

Dirty product

It is a very old herb and is not strictly a health food.

The herb is toxic to the body.

It is a serious health problem for the
---------------
Do you want to continue? (yes/no): no
```
...and there is some responce, not too poetic to be honest.

### Train and Tune modes
Before training or tuning the model, datasets needs to be prepared first.
```
python -m hello_gpt.datasets.txt2token_converter --input_path="hello_gpt/datasets/poems" --output_dir_path="hello_gpt/datasets/poems"
```
The `txt2token_converter` tool is used to load a file or a directory of text files and to output `train.bin` and `validate.bin` into a specified directory.

Once you have the `.bin` output, you might want to check if configuration parameters are to your liking in `datasets` directory. 
Configuration files follow the same philosophy: `model/train.yaml` has all of the basic configuration information, which gets updated (overriden) on specific fields specified with `--config` flag. Additionally you can specify a checkpoint to continue training from the previous point:
```
python -m hello_gpt.train.train --config="hello_gpt/datasets/poems/config_from_poe.yaml" --checkpoint="edgar_poe.pt"
```
Specifying or not specifying a checkpoint makes all the difference between `training` and `tuning`. Note however that tuning wasn't tested well for cases of model differences and you are likely to see a screen of errors if you would try to tune incompatible model. I'll do my best to fix this soon.

To tune pre-trained GPT2 models, all what you need to do is to have a tuning config and to use one of the supported by hugging face gpt models (gpt2, gpt2-medium, gpt2-large, gpt2-xl):
```
python -m hello_gpt.train.train --config="hello_gpt/datasets/poems/config_from_hug.yaml" --gpt2base="gpt2"
```
Note however, that larger models are likely to take more than 6GB of video memory.

There are two pretrained checkpoints already included with the project.
`edgar_poe.pt` is the model trained from scratch on laptop gpu for a few hours. It has a smaller custom configuration and achieved a loss of around `2` on that small training set.
`poems_hugs.pt` is the tuned on `poems` dataset smallest GPT2 model.

## License
This project is licensed under the terms of the LICENSE file.

The datasets were acquired thanks to the [Project Gutenberg](https://www.gutenberg.org/). 
Therefore texts of books to be distributed as per their license (long story short: recursively for free).
Please donate for their cause.