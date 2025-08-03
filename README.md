#  To run StoryCrafter training

```python
CUDA_VISIBLE_DEVICES=0 accelerate launch train_SC.py
```

# Inference StoryCrafter

```python
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_sc.py
```



# Code details

- config.yml: Configure the training parameters of the model.

- train_SC_stage1/2.py: Implement the training logics.

- inference_sc.py: Reasoning process of the model.

- model: Important structure of the model, include: 

  - attention.py: Implement the attention mechanism used in the model.

  - pipeline.py: Contain the pipeline code that orchestrates the entire process of the model

  - unet_2d_blocks.py: Define the 2D blocks used in the U-Net block of the model.

  - unet_2d_condition.py: Implement the conditional 2D U-Net model.

- test_image.py:  Inference on the test sets.

- dataset.py: Data processing.

- data: We provide the data storage path format, detailed data can download StorySalon dataset.
