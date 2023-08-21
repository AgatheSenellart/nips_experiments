"""
Store in this file all the shared variables for the comparison on mmnist.
"""

import argparse
import json

import torch
from torch import nn
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.base import EvaluatorConfig
from multivae.metrics.fids.fids import FIDEvaluator
from multivae.metrics.fids.fids_config import FIDEvaluatorConfig
from multivae.models import BaseMultiVAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from multivae.trainers import BaseTrainerConfig
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
import os

# The path to download and store the data, classifiers and inception network for FID.
data_path = './data'
output_path = 'experiments/'
use_wandb = False

modalities = ["m0", "m1", "m2", "m3", "m4"]

# Basic configuration informations shared accross models
base_config = dict(
    n_modalities=len(modalities),
    latent_dim=512,
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
    decoder_dist_params={k: {"scale": 0.75} for k in modalities},
)

# Define the encoders and decoders
encoders = {
    k: EncoderConvMMNIST_adapted(
        BaseAEConfig(
            latent_dim=base_config["latent_dim"], style_dim=0, input_dim=(3, 28, 28)
        )
    )
    for k in modalities
}

decoders = {
    k: DecoderConvMMNIST(
        BaseAEConfig(
            latent_dim=base_config["latent_dim"], style_dim=0, input_dim=(3, 28, 28)
        )
    )
    for k in modalities
}

# Define the training configuration
base_training_config = dict(
    learning_rate=1e-3,
    per_device_train_batch_size=256,
    num_epochs=800,
    optimizer_cls="Adam",
    optimizer_params={},
    steps_predict=5,
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 30},
)

wandb_project = "compare_on_mmnist"
config_name = "config2"


#######################################
## Define classifiers for the evaluation


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h


def load_mmnist_classifiers(data_path=os.path.join(data_path,'clf'), device="cuda"):
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


def eval_model(model, output_dir, test_data, wandb_path):
    """
    In this function, define all the evaluation metrics
    you want to use
    """
    config = CoherenceEvaluatorConfig(batch_size=512, wandb_path=wandb_path)

    # Cross-modal coherences
    CoherenceEvaluator(
        model=model,
        test_dataset=test_data,
        classifiers=load_mmnist_classifiers(device=model.device),
        output=output_dir,
        eval_config=config,
    ).eval()

    config = FIDEvaluatorConfig(batch_size=512, wandb_path=wandb_path,
                                inception_weights_path=os.path.join(data_path,'pt_inception-2015-12-05-6726825d.pth'))

    # Cross-modal FIDs
    FIDEvaluator(
        model, test_data, output=output_dir, eval_config=config
    ).compute_all_conditional_fids(gen_mod="m0")
