import argparse

from config2 import *

from multivae.models import JMVAE, JMVAEConfig

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

train_data = MMNISTDataset(
    data_path=data_path,
    split="train",
    missing_ratio=args.missing_ratio,
    keep_incomplete=args.keep_incomplete,
    download=True
)

test_data = MMNISTDataset(data_path=data_path, split="test", download=True)

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = JMVAEConfig(
    **base_config,
    alpha=0.1,
    warmup=200,
)

model = JMVAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    start_keep_best_epoch=model_config.warmup + 1,
    seed=args.seed,
    output_dir=f"{output_path}/compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/",
)

# Set up callbacks

if use_wandb:
    wandb_cb = WandbCallback()
    wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
    wandb_cb.run.config.update(args.__dict__)
    callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]
else:
    callbacks = None

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model

##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir, test_data, wandb_cb.run.path)
