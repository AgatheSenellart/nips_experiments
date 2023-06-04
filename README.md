

### For running experiments on MMNIST with partial data (with resnets architectures):

Create config_files with create_expes_files.py

And then run experiments with :

Slurm example
````

#!/bin/bash

#SBATCH --job-name=jmvae # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu               # GPU nodes are only available in gpu partition
#SBATCH --mem=30G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # output file name
#SBATCH --error=%x_%A_%a.out   # error file name


echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}

module purge
module load cuda/11.4.0

# Set your conda environment
source /home/$USER/.bashrc
# tensorflow environment shloud bre created previously
source activate multivaenv

wandb online

# For joint models : jmvae, jnf, jnfd
python expes/mmnist/training/resnets/jmvae.py --param_file config_only_incomplete/f${SLURM_ARRAY_TASK_ID}.json

# For aggregated models : mvtcae, mvae, mmvae, mmvae_plus, mopoe
python expes/mmnist/training/resnets/jmvae.py --param_file config/f${SLURM_ARRAY_TASK_ID}.json

```

