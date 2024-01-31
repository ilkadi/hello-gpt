# Hello-GPT architecture overview

In depth fabulous visualisation of GPT architecture: https://bbycroft.net/llm

Simplified meowalization for our readme.

```
Hello-GPT Model Structure (GPT2-based)
--------------------------------------
  ___________ 
 |           |       ____________________
 |   wte     |---->| Embedding Layer     |
 |___________|     | - Vocab Size        |
     |             | - Embedding Dim     |
     |             |_____________________|
     |                
     |             ____________________
     |            | Embedding Layer     |
     |----->| wpe | - Block Size        |
     |            | - Embedding Dim     |
     |            |_____________________|
     |
     |             ____________________
     |            | Dropout Layer       |
     |----->| drop| - Dropout Rate      |
     |            |_____________________|
     |
     |             ____________________       ____________________
     |            | Decoder Layer 1    |     | Decoder Layer N    |
     |            | (Variable Layers)  | ... | (N from Config)    |
     |----->|  h  | - Embedding Dim    |     | - Embedding Dim    |
     |            | - Number of Heads  |     | - Number of Heads  |
     |            | - Dropout          |     | - Dropout          |
     |            | - Bias             |     | - Bias             |
     |            |____________________|     |____________________|
     |
     |             ____________________
     |----->| ln_f| Normalisation      |
            |     | - Embedding Dim    |
            |     | - Bias             |
            |     |____________________|
            |
            |       ____________________
            |----->| Linear Layer       |
                   | - Output Vocab Size|
                   | - Embedding Dim    |
                   |____________________|

 * Note: wte.weight and lm_head.weight are shared.
```

# Directory overview:
```
Detailed Model Module Meowalization 
----------------------------------------------

+------------------------+
| model/ (Directory)     
| 'Purpose: Serves as a module housing the core components and initialization for HelloGPT.' 
+------------------------+
|
|-- model.py
|   'Purpose: Main file defining the HelloGPT model architecture.'
|   'Behavior: Acts as the root of the model module, integrating various components.'
|   'Relation: Utilizes decoder_transformer.py, mlp.py, normalisation.py, and self_attention.py.'
|
|-- model_initializer.py
|   'Purpose: Handles the initialization process of the HelloGPT model.'
|   'Behavior: Acts as an entry point for preparing the model with configurations.'
|   'Relation: Invokes model.py to create and configure the model instance.'
|
|-- abc_module.py
|   'Purpose: Defines Abstract Base Classes for consistent model structure.'
|   'Behavior: Provides foundational classes, but not directly involved in model architecture.'
|   'Relation: Serves as a building block for more complex components.'
|
|-- decoder_transformer.py
|   'Purpose: Implements the transformer decoder mechanism.'
|   'Behavior: Combines lower-level components like MLP and self-attention.'
|   'Relation: A core component used in model.py for decoding tasks.'
|
|-- mlp.py
|   'Purpose: Provides the Multi-Layer Perceptron functionality within the transformer.'
|   'Behavior: Acts as a leaf module, offering MLP capabilities to the decoder.'
|   'Relation: Integrated within decoder_transformer.py.'
|
|-- normalisation.py
|   'Purpose: Offers normalization layers for the model.'
|   'Behavior: Acts as a leaf module, directly utilized in model.py.'
|   'Relation: Essential for normalizing inputs/outputs within the model layers.'
|
|-- self_attention.py
|   'Purpose: Implements the self-attention mechanism for the model.'
|   'Behavior: Another leaf module, crucial for attention mechanisms in the transformer.'
|   'Relation: Used within decoder_transformer.py as a key component of the decoding process.'

* Notes for README:
  - The 'model' directory functions as a cohesive module, with 'model.py' at its core.
  - Each file has a specific role, either as a foundational component (like abc_module.py) or as a leaf module (like mlp.py or self_attention.py).
  - Understanding this hierarchical structure is key to grasping the model's design and functionality.
```