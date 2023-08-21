"""This file is used to create experiment configurations for the experiments on MHD"""

import itertools
import json
from pathlib import Path


folder = './configs_ms'
    
params_options = {
    "use_rescaling": [True, False],
    "beta": [0.5, 1, 2.5],
    "seed": [0, 1, 2, 3],
    
}

hypnames, hypvalues = zip(*params_options.items())
trial_hyperparameter_set = [
    dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
]

file_nb = 1
for i, config in enumerate(trial_hyperparameter_set):
    with open(f"{folder}/f{file_nb}.json", "w+") as fp:
        json.dump(config, fp)
    file_nb += 1
        
        
