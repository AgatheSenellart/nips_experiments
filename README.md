
This repository contains all the necessary files to reproduce the experiments presented in our 
paper *MultiVae : A Python library for reproducible and unified Multimodal Generative Autoencoders implementations*. 

## Setup

All you need is in our MultiVae package. You can install it by running:

```
git clone https://github.com/AgatheSenellart/multimodal_vaes.git
cd multimodal_vaes
pip install .
```


## Comparison on PolyMNIST

When running the comparison on PolyMNIST, you will have the following tree structure. 
The *data* folder will be downloaded automatically when running an experiment for the first time.
```
__nips_experiments
    |__comparison_on_mnist
        |__ config
        |__ config_only_incomplete
        |__ ...
    |__reproducing_results
        |__ ...
    |__data
        |__ MMNIST
        |__ clf
        |__ pt_inception-2015-12-05-6726825d.pth
```
### Training
        
The folders `config`and `config_only_incomplete` contains the information of the missing ratio parameter $(1-\eta)$, 
`keep_incomplete` variable and the seed. 

To launch the training of a model with the config contained in f1.json, move into the nips_experiments folder and run:

```bash
python comparison_on_mmnist/mvae.py --param_file comparison_on_mmnist/config/f1.json
```

You can change the model, and the configuration file. 
For aggregated models (mvae, mmvae, mopoe, mvtcae), all configurations in `config` are possible, for the joint encoder models (jmvae, jnf, jnfdcca) only the configuration in `config_only_incomplete` are possible, therefore the command you must use is :
```bash
python comparison_on_mmnist/jmvae.py --param_file comparison_on_mmnist/config_only_incomplete/f1.json
```


### Evaluate models and compute metrics 

To compute metrics with the pretrained models available on Hugging Face Hub, you will need to install huggingface :

```bash
pip install huggingface_hub
````

Move into the nips_experiments folder and run
```bash
python comparison_on_mmnist/eval_hf_models.py --param_file comparison_on_mmnist/config/f1.json --model_name MVAE
```

You can choose the configuration file and the name of the model, in order to download the chosen model from the Hub. 
Possible options for models : JMVAE, MMVAE, MVAE,MoPoE, MVTCAE, JNF, JNFDcca. Once again be careful that for joint encoder models, only the configurations in `config_only_incomplete`are possible.

You can also run the evaluation for a model trained on your own, using the `eval_local_models.py` script. 
You just need to provide the path to your trained model at the beginning of the file:
```python
model_path = "path_to_your_model"
```
and then run 
```bash
python comparison_on_mmnist/eval_local_models.py 
```

## Reproduce experiments 

### Setup
Make sure to download pretrained classifiers for evaluation whenever it is needed and place it in the `nips_experiments/data` folder.

|Dataset| Path to download classifiers|
|:--:|:--:|
|Mnist-SVHN| https://huggingface.co/asenella/mnist_svhn_classifiers/tree/main|
|MMNIST |https://zenodo.org/record/4899160#.ZGeXzy0isf_|


### How to run experiments ?

For each model, and experiment file and a validate file are available. 
In the validate.py file, by default a pretrained model is loaded from the HuggingFace Hub but you can provide the path to your locally trained model
instead. 
```python
model = AutoModel.load_from_hf_hub("asenella/reproduce_jmvae_seed_1", allow_pickle=True)
#### If you want to evaluate on your own trained model, uncomment the lines below
# model_path = enter_trained_model_path
# model = AutoModel.load_from_folder(model_path)
```

If you want to use wandb to monitor the experiments, you can uncomment the lines setting up the wandb callback in each file:
```python
# If you want to use wandb, uncomment the line below
callbacks=None
# wandb_ = WandbCallback()
# wandb_.setup(training_config, model_config, project_name="reproduce_jmvae")
# callbacks = [wandb_, ProgressBarCallback()]
```

In that case, you need to have wandb installed. 
```bash
pip install wandb
```

## MMVAE+ on partial data example

Move into the nips_experiments folder and run:
```bash 

python example_mmvae_plus/mmvae_plus.py --param_file comparison_on_mmnist/config/f1.json
```
You can change the configuration file number (f2.json, f3.json), to change the context of the experiments; the parameter $\eta$, the
`keep_incomplete` variable and the seed.

Once again, if you wish to use wandb, uncomment the following lines in the `mmvae_plus.py`file:

```python
##### Set up callbacks: Uncomment the following lines to use wandb
callbacks = None
# wandb_cb = WandbCallback()
# wandb_cb.setup(trainer_config, model_config)
# wandb_cb.run.config.update(args.__dict__)
# callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]
```