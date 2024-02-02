# Hello-GPT

Hello-GPT is a project that implements a simple version of the GPT model. 
It's designed to experiment with language models in a hands-on way.
While it was written for use on small personal machines, it does not require many modifications for multi-node multi-gpu execution settings.

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

## How to Run
With hello-gpt you can train your models from zero, tune those models on specific datasets and run them. 
You also can run pre-trained gpt2 hugging face models or finetune them on your datasets.
The sections below provide the basic out-of-the-box instructions for those functionalities.

Please keep in mind that the defaul configuration (`hello_gpt/train/train.yaml`) assumes the use of `cuda` device. If you don't have `cuda` installed, you might want to change device and device type to `cpu` in this config.

## Installation and testing

To use 'cuda' device, follow [NVIDIA Docs](https://developer.nvidia.com/cuda-downloads).
And install a correct version of [PyTorch](https://pytorch.org/).

To install the project's dependencies, run the following command:
```
pip install -r requirements.txt
```
Note that there are further optimisations available for newer GPUs but not implemented in this code.
To run unittests, from the root directory run:
```
python -m unittest discover tests
```

### Run mode
#### Hugging face models
The simplest possible way to run is to run with the pre-trained hugging face model. To do it, from the root directory run:
```
python -m hello_gpt.train.model_runner --gpt2base="gpt2"
```
This will download the base model (might take a bit of time) and start an interactive dialog in terminal. It is not a trained assistant, so user prompts simply provide a context for model to produce outputs. Supported hugging-face gpt2 models are (from the smallest to the biggest): `gpt2, gpt2-medium, gpt2-large, gpt2-xl`. Note that most of the project tests were run on `gpt2` model on laptop with `6Gb` VRAM.

#### Tuned hugging face models
Lets assume that you have a checkpoint for custom-tuned hugging face model (should be possible after you have a look into the training part of the readme). From the root directory run the following command:
```
python -m hello_gpt.train.model_runner --checkpoint="poems_hugs.pt" --gpt2base="gpt2"
```
Here we specify the checkpoint name (in `checkpoints` directory) and we unfortunately need to add a custom flag to notify the code that it is to follow a special model initialization process based on the hugging face implementation of GPT2.

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

#### Custom-trained models and configs
Under the hood, the `model_runner.py` script would load `train/train.yaml` config, update it with `model_run.yaml` contents and (optionaly) update with any other config supplied via `--config` flag. 

In this project you can find an example setup for the custom trained model: `datasets/edgar_poer/config_ext.yaml` specifies a model smaller than the gpt2. Assuming that you have followed the steps specified below to train the model, you can now run your very own model with the following command:
```
python -m hello_gpt.train.model_runner --checkpoint="edgar_poe.pt" --config="hello_gpt/datasets/edgar_poe/config_ext.yaml"
```
The script would load checkpoint stored on the root level of the project (`checkpoints` directory created on the same level as the `hello_gpt` and `tests` one) with the help of the config. Again an interactive console will appear:
```
Starting the interactive console..
>: hey
hey would have been beyond what is
      the essence in question.”

      “There are merely a genius,” says Le Soleil, “but it really
      may be in the reason possible reason.”

      “That is a good time,” said Dupin.

      “I have, of course; and I have, first, as it is, a common
      case, that this gentleman is a fool, I may, in a case, as well as
      an observation of the metaphysician; for to write at once, I shall
      attempt to bid you can above him again.”

      At length, looking towards him, and looking for breath I thought
      to keep him immediately
```
...but one night-long training on a laptop GPU probably won't give you that much of an output.

#### Custom-trained tuned models
In this repository you also can find sufficient resources to finetune model on a dataset. Assuming that you have a checkpoint of a fine-tuned model with a config of what it was fine-tuned on (`datasets/poems/confiig_from_poe.yaml` is the ready to use one) you can run it just like you would run model normally:
```
python -m hello_gpt.train.model_runner --checkpoint="poems_from_poe.pt" --config="hello_gpt/datasets/poems/config_from_poe.yaml"
```
and you can enjoy the output:
```
Starting the interactive console..
>: O Freedom!
O Freedom!
                                      To the
    the the the the
                                   And the, are its the
```

### Train and Tune modes
#### Convert datasets into bin format
Before training or tuning the model, datasets needs to be prepared first.
```
python -m hello_gpt.datasets.txt2token_converter --input_path="hello_gpt/datasets/poems" --output_dir_path="hello_gpt/datasets/poems"
```
The `txt2token_converter` tool is used to load a file or a directory of text files and to output `train.bin` and `validate.bin` into a specified directory.

#### Train from zero
Once you have the `.bin` output, you might want to check if configuration parameters are to your liking in `datasets` directory. 
Configuration files follow the same philosophy: `train/train.yaml` has all of the basic configuration information, which gets updated (overriden) on specific fields specified with `--config` flag. The below command assumes that `datasets/edgar_poe` has relevant `bin` files and utilisies the config to train the custom model:
```
python -m hello_gpt.train.model_trainer --config="hello_gpt/datasets/poems/config_from_poe.yaml"
```

#### Train from previous checkpoint
Additionally you can specify a checkpoint to continue training from the previous checkpoint:
```
python -m hello_gpt.train.model_trainer --config="hello_gpt/datasets/poems/config_from_poe.yaml" --checkpoint="edgar_poe.pt"
```
Specifying or not specifying a checkpoint makes all the difference between `training` and `tuning`. In other words it is pretty much the same process. Tuning simply loads model parameters from the file instead of having a random initialisation. 

#### Tune hugging face models
To tune pre-trained GPT2 models, all what you need to do is to have a tuning config and to use one of the supported by hugging face gpt models (gpt2, gpt2-medium, gpt2-large, gpt2-xl):
```
python -m hello_gpt.train.model_trainer --config="hello_gpt/datasets/poems/config_from_hug.yaml" --gpt2base="gpt2"
```
Note however, that larger models are likely to take more than 6GB of video memory.

## License
This project is licensed under the terms of the LICENSE file.

The datasets were acquired thanks to the [Project Gutenberg](https://www.gutenberg.org/). 
Therefore texts of books to be distributed as per their license (long story short: recursively for free).
Please donate for their cause.