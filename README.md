# MuRCL: Multi-instance Reinforcement Contrastive Learning for Whole Slide Image Classification
This repo is the PyTorch implementation for the MuRCL described in the paper "MuRCL: Multi-instance Reinforcement Contrastive Learning for Whole Slide Image Classification". 

![fig2](figs/fig2.png)

## Folder structures

```
│  train_MuRCL.py  # pre-training MuRCL
│  train_RLMIL.py  # training, fine-tuning and linear evaluating RLMIL 
│      
├─models
│      __init__.py
│      abmil.py
│      cl.py
│      clam.py
│      dsmil.py
│      rlmil.py
│      
└─utils
		__init__.py
        datasets.py  # WSI class and function for WSIs
        general.py   # help function
        losses.py    # loss function
        
```

## Requirements

TODO

## Datasets

### Download

TODO

### WSI Process

TODO

## Pre-training

pre-training our proposed framework MuRCL. 

```sh
cd runs
sh pretrain.sh
```

## Training from scratch, Fine-tuning, and linear evaluation

evaluation of our proposed framework MuRCL. 

```sh
cd runs
# training from scatch, does not load pre-trained weights
sh scratch.sh
# fine-tuning with pre-training weights by MuRCL
sh fintune.sh
# linear evaluation with pre-training weights by MuRCL
sh linear.sh
```



## Inference

TODO

## Visualization

TODO

## Training on your own datasets

TODO

