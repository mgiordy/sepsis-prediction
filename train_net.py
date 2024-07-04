# %%
import torch
import os
import sys
import yaml
from yaml.loader import SafeLoader
import wandb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torchinfo import summary

from helper import logging_conf
from helper.common import set_seed, parse_cmd_args

from helper.training_testing.dataset import get_dataloaders
from helper.training_testing import train

from models import tcn

logger = logging_conf.get_logger("main.py")


num_workers = int(os.cpu_count()//2)
logger.info(f"CPU count: {num_workers}")


if __name__ == '__main__':
    args = parse_cmd_args()

    # Read config and init Wandb
    config={}
    with open('network_config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Setting wandb mode
    if config["wandb"]:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Overwrite config with args
    for key, value in vars(args).items():
        if value is not None:
            logger.info(f"Overwriting config {key} with {value}")
            config[key] = value

    wandb.init(project=config["project_name"], tags=["train_torch", config["model"]], entity=config["entity"], config=config, allow_val_change=True)

    if len(wandb.config.datasets) == 0:
        logger.error("No dataset specified! Please specify at least one dataset to train on.")
        sys.exit(1)

    if not wandb.config.wandb:
        logger.warning("Running wandb in offline mode!!")

    # Setting seed for reproducibility
    set_seed(wandb.config.seed, logger)

    dtype = torch.float
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logger.info(f"Running on device: {device}")

    # Retrieving the right folder
    # Get the directory of the script, which is the upmost one
    current_script_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folders = {}
    for d in wandb.config.datasets:
        data_path = os.path.join(current_script_folder, "generated_datasets", d)
        if not os.path.exists(data_path):
            logger.error(f"Directory '{data_path}' does not exists! Generate dataset before training.")
            sys.exit(1)
        dataset_folders[d] = data_path

    datagen_params = {'batch_size': wandb.config.batch_size,
        'shuffle': True,
        'num_workers': num_workers}

    train_generator, test_generator = get_dataloaders(dataset_folders,
                                                      os.path.join(current_script_folder, "generated_datasets"),
                                                      wandb.config.k,
                                                      wandb.config.k_splits,
                                                      datagen_params,
                                                      custom_sampler=wandb.config.custom_sampler,
                                                      frequency=wandb.config.data_freq_min,
                                                      data_minutes=wandb.config.data_minutes,
                                                      prediction_minutes=wandb.config.prediction_minutes,
                                                      online_train=wandb.config.online_train,
                                                      online_test=wandb.config.online_test,
                                                      online_stride=wandb.config.online_stride,
                                                      time_to_sepsis_cutoff=wandb.config.time_to_sepsis_cutoff,
                                                      control_window_cutoff=wandb.config.control_window_cutoff,
                                                      col_to_keep=wandb.config.col_to_keep)


    # Importing the model
    if wandb.config.model == "tcn":
        model = tcn.MultiTCN(time_samples=wandb.config.data_minutes//wandb.config.data_freq_min, output_size=1, vital_signs=6, num_channels=wandb.config.num_channels, device=device, dtype=dtype, dense_layers=wandb.config.dense_layers, kernel_size=wandb.config.kernel_size, dropout=0.01, skip_conn=False, batch_norm=wandb.config.batch_norm, max_pooling=wandb.config.max_pooling)
    else:
        logger.error(f"Model {wandb.config.model} not found!")
        sys.exit(1)

    # Same here
    input_size = (1, 6, wandb.config.data_minutes//wandb.config.data_freq_min)
    summary(model.cpu(), input_size=input_size)

    loss = train.get_criterion()
    optimizer = train.get_optimizer(model, lr=wandb.config.lr, weight_decay=0)
    scheduler = train.get_scheduler(wandb.config.lr_scheduler, optimizer, len(train_generator), wandb.config.epochs, 1e-6, wandb.config.step_lr_epoch_div, wandb.config.step_lr_div_factor)

    model_save_path = os.path.join(current_script_folder, "trained_models", wandb.config.model + "_" + wandb.run.id)
    os.makedirs(model_save_path)
    train.train_nn(wandb, model_save_path, model, train_generator, test_generator, device, dtype, loss, optimizer, scheduler, wandb.config.epochs, len(train_generator), wandb.config.debug_grad, 1)

wandb.finish()