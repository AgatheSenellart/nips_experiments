
This repository contains all the necessary files to reproduce the experiments presented in our 
paper *MultiVae : A Python library for reproducible and unified Multimodal Generative Autoencoders implementations*. 

## Setup

All you need is in our MultiVae package. You can install it by running:

````
git clone https://github.com/AgatheSenellart/multimodal_vaes.git
cd multimodal_vaes
pip install .
```


## Comparison on PolyMNIST

insert tree

The folders `config`and `config_only_incomplete` contains the information of the missing ratio parameter $(1-\eta)$, 
`keep_incomplete` variable and the seed. 

To launch the training of a model with the config contained in f1.json, run (from inside the nips_experiments folder)

```bash
python comparison_on_mmnist/jmvae.py --param_file comparison_on_mmnist/config/f1.json
```

# Running evaluation on pretrained models


## Reproduce experiments 

Make sure to download pretrained classifiers for evaluation whenever it is needed.


|Dataset| Path to download classifiers|
|:--:|:--:|
|Mnist-SVHN| https://huggingface.co/asenella/mnist_svhn_classifiers/tree/main|
|MMNIST |https://zenodo.org/record/4899160#.ZGeXzy0isf_|


